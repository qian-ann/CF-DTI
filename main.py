
from models import DrugBAN
import torch.nn as nn
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import tempfile
import argparse
import warnings
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from multi_train_utils.distributed_utils import *
import random
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 也可以选其中的一个

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
# 临时存储,快照相关参数
parser.add_argument('--epochs_run', type=int, default=0)
parser.add_argument('--snapshot', type=bool, default=False)  # 是否采用存储的最好结果
parser.add_argument('--syncBN', type=bool, default=True)  # 是否同步BatchNorm
# 不要改该参数，系统会自动分配
parser.add_argument('--device', default='DDP', help='device id (i.e. cuda or cpu or DDP)')  #cuda表示可以直接运行
parser.add_argument('--DDP', default='True', help='whether DDP, False or True')
# parser.add_argument('--device', default='cuda', help='device id (i.e. cuda or cpu or DDP)')  #for debug
# parser.add_argument('--DDP', default='False', help='whether DDP, False or True') #for debug
parser.add_argument('--num_worker', type=int, default=8, help='num_worker')
parser.add_argument('--best_auroc', type=float, default=0, help='best_auroc')

parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random'])

# args = parser.parse_args(['--data','bindingdb','--split','random'])
args = parser.parse_args(['--data','biosnap','--split','random'])
# args = parser.parse_args(['--data','human','--split','random'])
# args = parser.parse_args(['--data','celegans','--split','random'])

# torchrun --standalone --nproc_per_node=2 main.py  #运行代码
# torchrun --standalone --nproc_per_node=1 main.py  #运行代码
# --standalone 表示在单机上进行训练

args_dir0 = str(args.data)+'_'+str(args.split)
cfg0 = get_cfg_defaults()
args.cfg = cfg0
Name = ''
idx = 1
if cfg0.modelT5:
    Name = Name + '_T5'
    if cfg0.noCNN:
        Name = Name + 'noCNN'
else:
    Name = Name + '_noT5'
if cfg0.Fe2:
    Name = Name + '_Fe2'
else:
    if cfg0.Fe1:
        Name = Name + '_Fe1'
if cfg0.noPE:
    Name = Name + '_noPE'
if cfg0.nofocal:
    Name = Name + '_nofocal'
if cfg0.modelmol:
    Name =Name +'_mol'

folder_path = cfg0.RESULT.OUTPUT_DIR + '//' + args_dir0 + fr'{Name}_{idx}'
while os.path.exists(folder_path):
    idx = idx + 1
    folder_path = cfg0.RESULT.OUTPUT_DIR + '//' + args_dir0 + fr'{Name}_{idx}'
args_dir = args_dir0 + fr'{Name}_{idx}'

softmax=True

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    if args.device == 'cpu' or args.device == 'cuda' or os.name != 'posix':
        print(args.device)
        args.device = torch.device(args.device)
        args.DDP=False
        rank=0
    else:
        # 初始化各进程环境
        init_distributed_mode(args=args)  #DDP
        rank = int(os.environ['LOCAL_RANK'])  # DDP
        cfg.SOLVER.LR *= int(os.environ['WORLD_SIZE'])  # 学习率要根据并行GPU的数量进行倍增 #DDP

    if cfg.SOLVER.SEEDMODE==0:
        seed=random.randint(1,10000)
        cfg.SOLVER.SEED=seed
    else:
        seed=cfg.SOLVER.SEED
    set_seed(seed)

    mkdir(cfg.RESULT.OUTPUT_DIR+'//'+args_dir)
    experiment = None
    # print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    if cfg.modelT5==True:
        train_dataT5 = h5py.File(dataFolder + '/train_dataT5.h5', 'r')
        val_dataT5 = h5py.File(dataFolder + '/val_dataT5.h5', 'r')
        test_dataT5 = h5py.File(dataFolder + '/test_dataT5.h5', 'r')
    else:
        train_dataT5 = None
        val_dataT5 = None
        test_dataT5 = None

    if cfg.modelmol==True:
        train_mol = h5py.File(dataFolder + '/train_mol.h5', 'r')
        val_mol = h5py.File(dataFolder + '/val_mol.h5', 'r')
        test_mol = h5py.File(dataFolder + '/test_mol.h5', 'r')
    else:
        train_mol = None
        val_mol = None
        test_mol = None

    train_dataset = DTIDataset(df_train.index.values, df_train, train_dataT5, train_mol, modelT5=cfg.modelT5, modelmol=cfg.modelmol)
    val_dataset = DTIDataset(df_val.index.values, df_val, val_dataT5, val_mol, modelT5=cfg.modelT5, modelmol=cfg.modelmol)
    test_dataset = DTIDataset(df_test.index.values, df_test, test_dataT5, test_mol, modelT5=cfg.modelT5, modelmol=cfg.modelmol)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE,
              'drop_last': True, 'collate_fn': graph_collate_func}
    batch_size=cfg.SOLVER.BATCH_SIZE

    '''
    多卡训练加载数据:
    # Dataset的设计上与单gpu一致，但是DataLoader上不一样。首先解释下原因：多gpu训练是，我们希望
    # 同一时刻在每个gpu上的数据是不一样的，这样相当于batch size扩大了N倍，因此起到了加速训练的作用。
    # 在DataLoader时，如何做到每个gpu上的数据是不一样的，且gpu1上训练过的数据如何确保接下来不被别
    # 的gpu再次训练。这时候就得需要DistributedSampler。
    # dataloader设置方式如下，注意shuffle与sampler是冲突的，并行训练需要设置sampler，此时务必
    # 要把shuffle设为False。但是这里shuffle=False并不意味着数据就不会乱序了，而是乱序的方式交给
    # sampler来控制，实质上数据仍是乱序的。
    '''
    if args.DDP:  #DDP
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        # 将样本索引每batch_size个元素组成一个list
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size,
                                                            drop_last=True)  # 验证集不需要这样  #DDP
        # number of workers #可以加快数据获取速度 #DDP
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, args.num_worker])
        if rank == 0:
            print('Using {} dataloader workers every process'.format(nw))  # 加载线程数 #DDP
        # 由于采用的sampler自带shuffle，这里需要是默认false
        training_generator = DataLoader(train_dataset,
                                       batch_sampler=train_batch_sampler,  # DDP
                                       pin_memory=True,  # DDP
                                       num_workers=nw,  collate_fn=params['collate_fn'])
        params['drop_last'] = False
        val_generator = DataLoader(val_dataset,
                                 sampler=val_sampler,  # 注意不是batchsampler #DDP
                                 pin_memory=True,
                                 num_workers=nw, **params)
        test_generator = DataLoader(test_dataset,
                                   sampler=test_sampler,  # 注意不是batchsampler #DDP
                                   pin_memory=True,
                                   num_workers=nw, **params)
    else:
        training_generator = DataLoader(train_dataset, shuffle=True, **params)
        params['drop_last'] = False
        val_generator = DataLoader(val_dataset, shuffle=False, **params)
        test_generator = DataLoader(test_dataset, shuffle=False, **params)

    model = DrugBAN(args.device,cfg).to(args.device)

    #将所有显卡初始模型参数统一start
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致 #DDP
    # 需要每个gpu上的权重一样 #DDP
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)
    if args.DDP:
        dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))  # DDP
    #将所有显卡初始模型参数统一end

    if args.syncBN and args.DDP:
        # 使用SyncBatchNorm后训练会更耗时 # 设置多个gpu的BN同步
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)  # DDP

    if args.DDP:
        print('use {} gpus!'.format(os.environ['WORLD_SIZE']))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpuid], find_unused_parameters=True) # 转为DDP模型 #DDP

    if args.snapshot:    #是否之前读取存储的最好模型结果 #DDP
        load_snapshot(model,args,args_dir,cfg)
        if args.DDP:
            dist.barrier()  # 等待所有的进程完成  #DDP

    pg = [p for p in model.parameters() if p.requires_grad]  # DDP
    opt = torch.optim.Adam(pg, lr=cfg.SOLVER.LR)  #注意pg #DDP
    scheduler = StepLR(opt, step_size=1, gamma=0.5)

    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, scheduler, args.device, training_generator, val_generator,
                      test_generator, args_dir, opt_da=None, discriminator=None,
                      experiment=experiment, epochs_run=args.epochs_run, rank=rank,
                      best_auroc=args.best_auroc, **cfg)

    if args.DDP:
        result = trainer.train(train_sampler,args)
    else:
        result = trainer.train(train_sampler=None,args=args)

    if rank==0:
        with open(os.path.join(cfg.RESULT.OUTPUT_DIR+'//'+args_dir, "model_architecture.txt"), "w") as wf:
            wf.write(str(model))
        print()
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
    cleanup()
