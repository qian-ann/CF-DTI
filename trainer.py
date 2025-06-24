import torch
import torch.nn as nn
import math
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm
from torch import distributed as dist
from multi_train_utils.distributed_utils import *


class Trainer(object):
    def __init__(self, model, optim, scheduler, device, train_dataloader, val_dataloader, test_dataloader, args_dir, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, epochs_run=0, rank=0, best_auroc=0, **config):
        self.model = model
        self.optim = optim
        self.best_auroc = best_auroc
        self.scheduler = scheduler
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = epochs_run
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["MLP"]["BINARY"]
        self.keyy = 0

        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.epochs_run = epochs_run #DDP
        self.rank=rank #DDP

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config

        self.output_dir = config["RESULT"]["OUTPUT_DIR"]+'//'+args_dir

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)



    def train(self,train_sampler=None,args=None):
        float2str = lambda x: '%0.4f' % x
        key_stop = 0
        for i in range(self.epochs_run,self.epochs):
            self.current_epoch += 1
            if args!=None and args.DDP:
                train_sampler.set_epoch(i)  # 这句莫忘，否则相当于没有shuffle数据  #DDP
            train_loss = self.train_epoch(args.DDP)  #DDP
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            auroc, auprc, val_loss = self.test(dataloader="val", DDP=args.DDP)

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            try:
                print('Check distribution '+"%.d" % int(os.environ['LOCAL_RANK'])+': auroc'+"%.3f" % auroc)
            except:
                print('Check distribution ' + ': auroc' + "%.3f" % auroc)
            if auroc >= self.best_auroc:
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                if self.config["RESULT"]["SAVE_MODEL"]:
                    if self.rank == 0:  # 只在主进程执行下面的寻找最好模型的操作 #DDP
                        snapshot = {
                            "MODEL_STATE": self.model.state_dict(),
                            "EPOCHS_RUN": i,
                            'BEST_AUROC': self.best_auroc,

                        }
                        torch.save(snapshot, os.path.join(self.output_dir, f"best_model.pth"))
                        print(f"Epoch {i+1} | Training best_model saved at {self.output_dir}")
                self.keyy = 0
            elif auroc < self.best_auroc:
                self.keyy += 1
                if self.keyy >= 10:
                    # checkpoint = torch.load(os.path.join(self.output_dir, f"best_model.pth"), map_location=self.device)
                    # self.model.load_state_dict(checkpoint["MODEL_STATE"])
                    self.scheduler.step()
                    key_stop += 1
                    self.keyy = 0
            if self.rank == 0:  # 只在主进程执行下面的寻找最好模型的操作 #DDP
                print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                      + str(auroc) + " AUPRC " + str(auprc))
                print(self.optim.param_groups[0]['lr'], self.keyy, key_stop, self.best_auroc)
                # if self.optim.param_groups[0]['lr'] < self.config["SOLVER"]["LR"] * math.pow(0.5, 3):
            if key_stop > 2:
                print("The best AUC is", "%.3f" % self.best_auroc)
                break
        if args.DDP:
            dist.barrier()  # 这一句作用是：所有进程(gpu)上的代码都执行到这，才会执行该句下面的代码
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss,  \
            thred_optim, precision, att1s, att2s, att3s, attidxds, attidxps, y_label, y_pred= self.test(dataloader="test", DDP=args.DDP)

        # np.save(self.output_dir + '/att1s', att1s)
        # np.save(self.output_dir + '/att2s', att2s)
        # np.save(self.output_dir + '/att3s', att3s)
        # np.save(self.output_dir + '/attidxds', attidxds)
        # np.save(self.output_dir + '/attidxps', attidxps)
        np.save(self.output_dir + '/y_label', y_label)
        np.save(self.output_dir + '/y_pred', y_pred)

        if self.rank == 0:  # 只在主进程执行下面的寻找最好模型的操作 #DDP
            test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                                accuracy, thred_optim, test_loss]))
            self.test_table.add_row(test_lst)
            print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
                  str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
            self.test_metrics["auroc"] = auroc
            self.test_metrics["auprc"] = auprc
            self.test_metrics["test_loss"] = test_loss
            self.test_metrics["sensitivity"] = sensitivity
            self.test_metrics["specificity"] = specificity
            self.test_metrics["accuracy"] = accuracy
            self.test_metrics["thred_optim"] = thred_optim
            self.test_metrics["best_epoch"] = self.best_epoch
            self.test_metrics["F1"] = f1
            self.test_metrics["Precision"] = precision
            self.save_result()

        return self.test_metrics

    def save_result(self):
        float2str = lambda x: '%0.4f' % x
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        seed=self.config["SOLVER"]["SEED"]
        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, f"train_markdowntable.txt")
        seed_file = os.path.join(self.output_dir, f"seed.txt")
        val_lst = ["Best epoch " + str(self.best_epoch)] + list(map(float2str, [self.best_auroc, self.best_auroc, self.best_auroc]))
        self.val_table.add_row(val_lst)
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())
        with open(seed_file, "w") as fp:
            fp.write(f'{seed}')


    def train_epoch(self,DDP):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)

        for i, (v_d, v_p, labels, mol_vec) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels, mol_vec = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), mol_vec.to(self.device)
            self.optim.zero_grad()
            score = self.model(v_d, v_p, mol_vec)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            if DDP:
                dist.barrier()  # 等待所有的进程完成  #DDP
                loss=reduce_value(loss,average=True)
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        if self.rank==0:
            print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        if DDP:
            dist.barrier()  # 等待所有的进程完成  #DDP
        return loss_epoch

    def test(self, dataloader="test", DDP=None):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels, mol_vec) in enumerate(data_loader):
                v_d, v_p, labels, mol_vec = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), mol_vec.to(self.device)

                if dataloader == "val":
                    score, att1, att2, att3, attidxd, attidxp = self.model(v_d, v_p, mol_vec, mode='val')
                elif dataloader == "test":
                    checkpoint = torch.load(os.path.join(self.output_dir, f"best_model.pth"), map_location=self.device)
                    self.model.load_state_dict(checkpoint["MODEL_STATE"])
                    if DDP:
                        dist.barrier()  #等待所有的进程完成  #DDP
                    score, att1, att2, att3, attidxd, attidxp = self.model(v_d, v_p, mol_vec, mode='test')
                    if i==0:
                        if att1 == None:
                            att1s = 0
                        else:
                            att1s = np.array(att1.cpu())
                        if att2 == None:
                            att2s = 0
                        else:
                            att2s = np.array(att2.cpu())
                        if att3 == None:
                            att3s = 0
                        else:
                            att3s = np.array(att3.cpu())
                        if attidxd == None:
                            attidxds = 0
                        else:
                            attidxds = np.array(attidxd.cpu())
                        if attidxp == None:
                            attidxps = 0
                        else:
                            attidxps = np.array(attidxp.cpu())
                    # else:
                    #     atts = np.concatenate([atts,np.array(att.cpu())],axis=0)
                    #     att1s = np.concatenate([att1s,np.array(att1.cpu())],axis=0)
                    #     att2s = np.concatenate([att2s,np.array(att2.cpu())],axis=0)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                if DDP:  #DDP
                    loss = reduce_value(loss, average=True)  # 数据用gather #DDP
                    n = gather_value(n, self.device)
                    labels = gather_value(labels, self.device)
                    dist.barrier()
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, \
                   thred_optim, precision1, att1s, att2s, att3s, attidxds,  attidxps, y_label, y_pred
        else:
            return auroc, auprc, test_loss
