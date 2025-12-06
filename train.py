import os,sys,logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models
from torch.autograd import Variable
from PIL import ImageFile
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import jensenshannon

import config
from config import _ParamConfig
from config_sp import Config_SP
from config_nass import Config_NASS
from sampling_strategy.strategy_factory import BatchSampleSelectionStratgyFactory
from utils import *
from model import TimeSeriesCNN
from datasets import MultivariateTSDataset

import argparse

from losses import *

parser = argparse.ArgumentParser(description="Part of parameters settings for Basicconfig and SP config")

parser.add_argument('--T', help='时间序列长度', choices=[60,120,180],default=180)
parser.add_argument('--B', help='每一轮的查询预算', choices=[10,30,50,70,90,100],default=50)
parser.add_argument('--method', help='价值度量方法', default="lebgp-margin", choices=["random", "entropy", "leastconfidence","margin", "vr", "emc",
                                                                          "lebgp-entropy", "lebgp-leastconfidence", "lebgp-margin"])
parser.add_argument('--nass', help='是否用nass再平衡', choices=[True,False], default=True)
parser.add_argument('-gamma', help="邻域半径", choices=[0, 0.01, 0.05, 0.1], default=0.01)
parser.add_argument('-_lambda', help="似然权重", choices=[1,5,10,20,100,"dym"], default="dym")
parser.add_argument('--loss', help='损失函数', choices=["ce", "twl", "fl", "cbl", "eql", "cdbl", "dqbl"], default="ce")
args = parser.parse_args()

def main():
    exp_count = 1
    CONFSP = Config_SP(T=args.T)
    ds = MultivariateTSDataset(CONFSP)
    train_ds, test_ds = MultivariateTSDataset.gen_train_test_ds(ds)
    # for nass in [False, True]:
    #     for B in [30, 50, 70, 90]:
    #         for gamma in [0, 0.01, 0.05, 0.1]: # 0 indicates do not using neighborhood correction
    #             for _lambda in [1, 5, 10, 20, 100, "dym"]:
    #                 if nass:
    #                     methods = ["random", "entropy", "leastconfidence","margin", 
    #                                 "lebgp-entropy", "lebgp-leastconfidence", "lebgp-margin"]
    #                 else:
    #                     methods = ["lebgp-entropy", "lebgp-leastconfidence", "lebgp-margin"]
    #                 for method in methods:
    CONFNASS = Config_NASS(gamma=args.gamma, _lambda=args._lambda)
    CONF = _ParamConfig(B=args.B, method=args.method, nass=args.nass, loss="ce", CONFSP=CONFSP, CONFNASS=CONFNASS)
    CONF.logger.critical(msg=f"NASS:{args.nass}, T:{180}, method:{args.method}")
    set_seed(CONF.seed)
    CONF.logger.critical(msg=str(CONF))
    CONF.logger.critical(msg="Load Model")
    model = TimeSeriesCNN(CONFSP.n_features,CONFSP.n_classes)
    classifier = model.classifier

    model.to(CONF.DEVICE)
    classifier.to(CONF.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONF.lr)

    train_comp(model, classifier, train_ds, test_ds, 
            **{"feature_maps": None, "optimizer": optimizer,"CONF":CONF,"CONFSP":CONFSP,
                "gamma":args.gamma,"_lambda":args._lambda})
    
    CONF.logger.info(msg=f"Experimental ID: {exp_count}/960")
    exp_count += 1
    cleanup_logger("Log: ")

# One AL round training
def one_al_train(round, model, classifier, optimizer, labeled_ds, unlabeled_ds, n_epochs, CONF):
    CONF.logger.critical("Training")
    model.train()
    classifier.train()
    CONF.logger.info(f"Labeled dataset size: {len(labeled_ds)}, unlabeled dataset size: {len(unlabeled_ds)}")
    # We use cross-training, labeled batch-->unlabeled batch-->labeled batch-->...
    for i in range(n_epochs):
        labeled_ds_loader = torch.utils.data.DataLoader(labeled_ds, batch_size=CONF.bs, shuffle=True)
        labeled_ds_loader_ite = iter(labeled_ds_loader)
        x_batch, y_batch = None, None
        __iflabeled__ = False
        data_length = len(labeled_ds_loader)
        sup_loss_total,rec_loss_total = 0,0
        for j in range(data_length):
            x_batch, y_batch = next(labeled_ds_loader_ite)
            x_batch, y_batch = x_batch.to(CONF.DEVICE), y_batch.to(CONF.DEVICE)
            #y_batch = torch.squeeze(y_batch, dim=1)
            z = model(x_batch)
            logits = classifier(z)
            t_loss = CONF.loss(logits, y_batch)
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            sup_loss_value = t_loss.data.item()
            sup_loss_total += sup_loss_value

            if (i+1) % 100 == 0:
                CONF.logger.info(f"Epoch: {i+1}/{n_epochs}, batch/total: {j}/{data_length*2} Supervised Loss Value: {sup_loss_value:.4f}, Reconstructive Loss Value (/w_i): {-1.0:.4f}")            
        
        f = open(CONF.train_loss_file, "a")
        #f.write(f"{round},{i+1},{sup_loss_total/data_length:.4f},{rec_loss_total/data_length:.4f},{sup_loss_total/data_length + current_w * rec_loss_total/data_length:.4f},{current_w:.4f}")
        f.write(f"{round},{i+1},{sup_loss_total/data_length:.4f}")
        f.write("\n")
        f.close()

def test(round, model, classifier, test_ds, CONF, CONFSP):
    CONF.logger.critical("Validating")
    model.eval()
    classifier.eval()
    CONF.logger.info(f"Test dataset size: {len(test_ds)}")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=CONF.bs, shuffle=False)
    pred_labels = []
    gt_labels = []
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(CONF.DEVICE), y_batch.to(CONF.DEVICE)
            z = model(x_batch)
            logits = classifier(z)
            probs = F.softmax(logits, dim=1)
            _, pred = torch.max(probs, dim=1)
            pred_labels.extend(list(pred.cpu().numpy()))
            gt_labels.extend(list(y_batch.cpu().numpy()))
            if i % 50 == 0:
                CONF.logger.info(f"Batch: {i}/{len(test_loader)}")
        
        pred_labels = np.asarray(pred_labels).astype(int)
        gt_labels = np.asarray(gt_labels).astype(int)
        #acc = accuracy_score(gt_labels, pred_labels)
        pred_res = weighted_prec_rec_f1_detailed(gt_labels, pred_labels, CONFSP.population_ratio,CONF=CONF,CONFSP=CONFSP)
        WP = pred_res["weighted_precision"]
        WR = pred_res["weighted_recall"]
        WF = pred_res["weighted_f1"]
        WP_NW = pred_res["weighted_precision_nw"]
        WR_NW = pred_res["weighted_recall_nw"]
        WF_NW = pred_res["weighted_f1_nw"]
        CONF.logger.info(f"Validation WP/WR/WF/WP1/WR1/WF1: {WP * 100:.2f}%/{WR * 100:.2f}%/{WF * 100:.2f}%/{WP_NW * 100:.2f}%/{WR_NW * 100:.2f}%/{WF_NW * 100:.2f}%")

        class_wise_accuracy = np.asarray(pred_res["per_class_recall"])
        if CONF.loss_name == "cdbl":
            CONF.loss = CDB_loss(class_difficulty = 1 - class_wise_accuracy, tau = "dynamic", reduction="mean")
        elif CONF.loss_name == "dqbl":
            acc, entropy = CONF.loss.computeAccEntropy(model, classifier, test_loader)
            CONF.loss.update_DQBWeight(acc, entropy)
    f = open(CONF.val_acc_file, "a")
    f.write("{},{:.2f},{:.2f},{:.2f}\n".format(round, WP*100, WR*100, WF*100))
    f.close()
    return WF

# AL整体训练
def train_comp(model, classifier, train_ds, test_ds, **kwargs):
    # Step 1.0 数据集划分为初始标签集合和无标签池
    labeled_idx = []
    CONF = kwargs["CONF"]
    CONFSP = kwargs["CONFSP"]
    class_counts = [0] * CONF.n_classes
    for i in range(len(train_ds)):
        x, y = train_ds[i]
        if class_counts[y.item()] < CONF.init_labels / CONF.n_classes:
            class_counts[y.item()] += 1
            labeled_idx.append(i)

    CONF.logger.info(class_counts)
    labeled_idx = np.asarray(labeled_idx).astype(int)
    labeled_ds = torch.utils.data.Subset(train_ds, labeled_idx)
    unlabeled_idx = list(set([i for i in range(len(train_ds))]).difference(labeled_idx))
    unlabeled_ds = torch.utils.data.Subset(train_ds, np.asarray(unlabeled_idx,dtype=int))
    n_cs = true_sample_idx_each_class(trainset=train_ds, _labeled_idx=labeled_idx,CONF=CONF)
    n_cs = [len(x) for x in n_cs]
    f = open(CONF.nc_file, "a")
    f.write("{},{}\n".format(0,",".join([str(x) for x in n_cs])))
    f.close()

    # Step 2.0 初始训练
    CONF.logger.info("Initial training samples ID: " + str(labeled_idx))
    one_al_train(0, model,classifier,kwargs["optimizer"],labeled_ds,unlabeled_ds,CONF.E2, CONF)
    CONF.logger.info("Initial Training Completed")
    
    # Step 2.1 选择样本查询策略 gass和non-gass
    best_acc = test(0, model, classifier, test_ds, CONF, CONFSP)
    for c_round in range(1, CONF.E1): # 查询轮数
        CONF.logger.info(f"Sampling Method: {CONF.method}, Using gass: {CONF.gass}, Known population:{CONF.known_population}")
        added_idx = _sampling_(model, classifier, train_ds, labeled_idx, **{**kwargs, "round": c_round})
        labeled_idx = np.concatenate([labeled_idx, added_idx]).astype(int)
        labeled_ds = torch.utils.data.Subset(train_ds, labeled_idx)
        unlabeled_idx = list(set([i for i in range(len(train_ds))]).difference(labeled_idx))
        unlabeled_ds = torch.utils.data.Subset(train_ds, np.asarray(unlabeled_idx, dtype=int))
        
        # 获取每个类别的真实样本数量
        n_cs = true_sample_idx_each_class(trainset=train_ds, _labeled_idx=labeled_idx,CONF=CONF)
        n_cs = [len(x) for x in n_cs]
        CONF.logger.info("{}th Round. The Queried Sample Size n_cs:{}.".format(c_round, str(n_cs)))
        
        f = open(CONF.nc_file, "a")
        f.write("{},{}\n".format(c_round,",".join([str(x) for x in n_cs])))
        f.close()

        phat = np.asarray(n_cs) / np.sum(n_cs)
        population = np.asarray(CONFSP.population_ratio)
        js = jensenshannon(phat, population)
        f = open(CONF.js_file, "a")
        f.write("{},{}\n".format(c_round,js))
        f.close()

        # 训练
        #current_w = CONF.w_max - c_round * (CONF.w_max - CONF.w_min) / (CONF.E1 - 1)
        if CONF.dynamic_E2:
            E2 = int(np.ceil(c_round * CONF.E2 / (CONF.E1-1)))
        one_al_train(c_round, model, classifier, kwargs["optimizer"], labeled_ds, unlabeled_ds, E2, CONF)
        acc = test(c_round, model, classifier, test_ds, CONF, CONFSP)
        if acc > best_acc:
            best_acc = acc
            # 实验较多多，不存模型参数
            #torch.save(model.state_dict(), CONF.model_file)

def _sampling_(model, classifier, train_ds, labeled_idx, **kwargs):
    strategy = None
    CONF = kwargs["CONF"]
    CONFSP = kwargs["CONFSP"]

    if CONF.gass:
        strategy = BatchSampleSelectionStratgyFactory.\
                    get_gass_strategy(CONF.method,train_ds,labeled_idx,budget=CONF.B,\
                                 **{"model":model, "classifier":classifier, "population_ratio": CONFSP.population_ratio, **kwargs})
    elif CONF.nass:
        strategy = BatchSampleSelectionStratgyFactory.\
                    get_nass_strategy(CONF.method,train_ds,labeled_idx,budget=CONF.B,\
                                 **{"model":model, "classifier":classifier, "population_ratio": CONFSP.population_ratio, **kwargs})
    else:
        strategy = BatchSampleSelectionStratgyFactory.\
                    get_strategy(CONF.method,train_ds,labeled_idx,budget=CONF.B,\
                                 **{"model":model, \
                                    "classifier":classifier,
                                     **kwargs})

    return strategy.select()

if __name__ == "__main__":
    main()
