import os

import torch
import torch.distributed as dist


def init_distributed_mode(args):  #DDP
    args.gpuid = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.gpuid)
    args.device = torch.device(args.gpuid)
    print('| distributed init (rank {})'.format(os.environ['LOCAL_RANK']), flush=True)
    dist.init_process_group(backend='nccl')  # 通信后端，nvidia GPU推荐使用NCCL
    dist.barrier()  # DDP


def load_snapshot(model,args,args_dir=None,cfg=None):   #DDP
    print(args.cfg)
    output_dir = cfg["RESULT"]["OUTPUT_DIR"] + '//' + args_dir
    if os.path.exists(os.path.join(output_dir, f"best_model.pth")):
        checkpoint = torch.load(os.path.join(output_dir, f"best_model.pth"), map_location=args.device)
        model.load_state_dict(checkpoint["MODEL_STATE"])
        args.epochs_run = checkpoint["EPOCHS_RUN"]
        args.best_auroc = checkpoint["BEST_AUROC"]
        print(f"Resuming training from best_model at Epoch {args.epochs_run+1}")

def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

def gather_value(value,device):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
    tensor_list = [torch.zeros_like(value).to(device) for _ in range(world_size)]
    with torch.no_grad():
        dist.all_gather(tensor_list, value)
        value = torch.cat(tensor_list, dim=0)

        return value
