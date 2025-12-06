import sys

import torch.utils
import torch.utils
from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch
from complexity1 import *
from utils import pred_probs,compute_uncertainty,stratify,true_sample_idx_each_class,compute_bc,compute_bc_knownpopulation
# Renormalization group multiscale complexity
class GASSExplorationPStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]
        self.round = kwargs["round"]
        

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"Exploration-P Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities and entropies")
        self.model.eval()
        self.classifier.eval()
        
        C = CONF.n_classes
        query_id = self.round
        q_budget = CONF.B
        q_round = CONF.E1 - 1
        bs = CONF.bs
        beta = 1e-2
        exploitation_m = int(q_budget * query_id / q_round)
        n_samples = len(self.train_ds)
        remain_probs = pred_probs(self.model, self.train_ds, self.unlabeled_idx,CONF)
        pred_C_idx = stratify(remain_probs,CONF)
        true_sample_idx = true_sample_idx_each_class(trainset=self.train_ds, _labeled_idx=self.labeled_idx,CONF=CONF)
        n_cs = [len(x) for x in true_sample_idx]
        b_cs = compute_bc(n_cs,CONF)
        CONF.logger.info("The sample size of each class (Model Predicted): {}".format(str([len(x) for x in pred_C_idx])))
        CONF.logger.info("Optimal query sample size: bc: {}".format(str(b_cs)))
        b_c_exploitation = [int(x * query_id / q_round) for x in b_cs]
        b_c_exploration = [b_cs[i] - b_c_exploitation[i] for i in range(C)]
        CONF.logger.info("Optimal query sample size (Exploitation Stage): bc: {}".format(str(b_c_exploitation)))
        CONF.logger.info("Optimal query sample size (Exploration Stage): bc: {}".format(str(b_c_exploration)))
        remain_probs = pred_probs(self.model,self.train_ds,self.unlabeled_idx)
        remain_entropy = compute_uncertainty(remain_probs, "entropy")
        remain_id_feature_mapping = {}
        # 存储所有样本的特征矩阵，特征的维度是512
        print("Compute the feature matrix of labeled/unlabeled samples")
        c_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=bs, shuffle=False, num_workers=0)
        with torch.no_grad():
            for i, (x, _) in enumerate(c_loader):
                features = self.model(x.to(CONF.DEVICE))
                for j in range(i*bs, min((i+1)*bs, len(self.train_ds))):
                    remain_id_feature_mapping[j] = features[j - i*bs]
                    
        SS_set = [[] for _ in range(C)]
        for i in range(C):
            print(f"\r Class {i}/{C}", end="", flush=True)
            if b_cs[i] == 0: continue
            # Step 2. 计算每个类（预测）中最大的熵值所对应的样本
            bc_exploit_c = b_c_exploitation[i]
            bc_exploration_c = b_c_exploration[i]
            pred_c_idx = pred_C_idx[i]
            if len(pred_c_idx) == 0:
                continue
            enc_vec = np.asarray([remain_entropy[j] for j in pred_c_idx])
            t_SS_set = {pred_c_idx[np.argmax(enc_vec)]}
            t_U_set = set(pred_c_idx).difference(t_SS_set)
            GAP = 400
            print("Exploitation Stage ")
            for count in range(bc_exploit_c - 1):
                U_diff_S = list(t_U_set.difference(t_SS_set))
                if len(U_diff_S) == 0:
                    break
                S_fmat = torch.stack([remain_id_feature_mapping[id] for id in t_SS_set]).T
                ent_vec = torch.tensor([remain_entropy[id] for id in U_diff_S]).to(CONF.DEVICE)
                n = len(U_diff_S)
                max_info = -sys.maxsize
                sel_ind = sys.maxsize
                for j in range(0, n, GAP):
                    start = j
                    end = min(j+GAP, n)
                    U_diff_S_fmat = torch.stack([remain_id_feature_mapping[id] for id in U_diff_S[start:end]])
                    tt_mat = torch.matmul(U_diff_S_fmat, S_fmat)
                    II = ent_vec[start:end] - beta * torch.max(tt_mat, axis=1)[0]
                    can_ind = torch.argmax(II).item()
                    if II[can_ind].item() > max_info:
                        max_info = II[can_ind].item()
                        sel_ind = can_ind + start
                torch.cuda.empty_cache()
                t_SS_set.add(U_diff_S[int(sel_ind)])

            print("\nExploration Stage   ")
            GAP = 200
            for count in range(b_cs[i] - len(t_SS_set)):
                U_diff_S = list(t_U_set.difference(t_SS_set))
                if len(U_diff_S) == 0:
                    break
                L_cup_S = list(self.labeled_idx) + list(t_SS_set)
                L_cup_S_fmat = torch.stack([remain_id_feature_mapping[id] for id in L_cup_S]).T.to(CONF.DEVICE)
                n = len(U_diff_S)
                max_info = -sys.maxsize
                sel_ind = sys.maxsize
                for j in range(0, n, GAP):
                    start = j
                    end = min(j+GAP, n)
                    U_diff_S_fmat = torch.stack([remain_id_feature_mapping[id] for id in U_diff_S[start:end]])
                    tt_mat = torch.matmul(U_diff_S_fmat, L_cup_S_fmat)
                    II = -torch.max(tt_mat, axis=1)[0]
                    can_ind = torch.argmax(II).item()
                    if II[can_ind].item() > max_info:
                        max_info = II[can_ind].item()
                        sel_ind = can_ind + start
                torch.cuda.empty_cache()
                t_SS_set.add(U_diff_S[int(sel_ind)])
                print("\r {}/{}".format(count, bc_exploration_c), end="", flush=True)
            print()
            SS_set[i] = t_SS_set
        
        SS_list = [list(x) for x in SS_set]
        budgeted = np.sum([len(x) for x in SS_list])
        CONF.logger.info("Query Budget: {}, Satisfied：{}".format(q_budget, budgeted))
        assert budgeted <= q_budget
        adequates = [True] * C
        for i in range(C):
            # 如果第i个类别的样本数小于该类别的预算，则设置adequate变量为False
            adequates[i] = len(pred_C_idx[i]) >= b_cs[i]
        class_id = -1
        if budgeted < q_budget:
            CONF.logger.warning("Polling Supply.")
        while budgeted < q_budget: # 采用轮询的方法抽取剩余样本    
            class_id = (class_id + 1) % C
            if adequates[class_id]:
                rem_idx = set(pred_C_idx[class_id]).difference(set(SS_list[class_id]))
                rem_idx = list(rem_idx)
                if len(rem_idx) == 0:
                    adequates[class_id] = False
                    continue
                id = np.random.choice(rem_idx)
                SS_list[class_id].append(id)
                budgeted += 1
        SS_list = np.concatenate(SS_list).astype(int)
        return SS_list