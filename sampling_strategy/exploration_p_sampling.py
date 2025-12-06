import sys

import torch.utils
from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch
from complexity1 import *
from utils import pred_probs,compute_uncertainty
# Renormalization group multiscale complexity
class ExplorationPStrategy(BatchSampleSelectionStratgy):
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
        
        query_id = self.round
        q_budget = CONF.B
        q_round = CONF.E1 - 1
        remain_idx = set(np.arange(len(self.train_ds), dtype=np.int32)).difference(set(self.labeled_idx))
        remain_idx = list(remain_idx)
        # the number of eploitation 
        beta = 1e-2
        exploitation_m = int(q_budget * query_id / q_round)
        remain_probs = pred_probs(self.model, self.train_ds, self.unlabeled_idx)
        remain_entropy = compute_uncertainty(remain_probs, "entropy")
        remain_id_feature_mapping = {}
        # 存储所有样本的特征矩阵，特征的维度是512
        #print("计算已标注样本和未标注样本的特征矩阵")
        print("Compute the feature matrix of labeled/unlabeled samples")
        bs = CONF.bs
        c_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=bs, shuffle=False, num_workers=0)
        with torch.no_grad():
            for i, (x, _) in enumerate(c_loader):
                features = self.model(x.to(CONF.DEVICE))
                for j in range(i*bs, min((i+1)*bs, len(self.train_ds))):
                    remain_id_feature_mapping[j] = features[j - i*bs]
                
        max_ent_id = sorted(remain_entropy.items(), key=lambda x: x[1])[-1][0]
        S_set = {max_ent_id}
        U_set = set(remain_idx)
        with torch.no_grad():
            print("Exploitation Stage   ")
            GAP = 200
            for count in range(exploitation_m):
                U_diff_S = list(U_set.difference(S_set))
                S_fmat = torch.stack([remain_id_feature_mapping[id] for id in S_set]).T
                ent_vec = torch.tensor([remain_entropy[id] for id in U_diff_S]).to(CONF.DEVICE)
                n = len(U_diff_S)
                max_info = -sys.maxsize
                sel_ind = sys.maxsize
                for i in range(0, n, GAP):
                    start = i
                    end = min(i+GAP, n)
                    U_diff_S_fmat = torch.stack([remain_id_feature_mapping[id] for id in U_diff_S[start:end]])
                    tt_mat = torch.matmul(U_diff_S_fmat, S_fmat)
                    II = ent_vec[start:end] - beta * torch.max(tt_mat, axis=1)[0]
                    can_ind = torch.argmax(II).item()
                    if II[can_ind].item() > max_info:
                        max_info = II[can_ind].item()
                        sel_ind = can_ind + start
                torch.cuda.empty_cache()
                S_set.add(U_diff_S[int(sel_ind)])
                print("\r {}/{}".format(count, exploitation_m), end="", flush=True)
                
            print("\nExploration Stage   ")
            GAP = 50
            for count in range(q_budget - exploitation_m - 1):
                U_diff_S = list(U_set.difference(S_set))
                L_cup_S = list(self.labeled_idx) + list(S_set)
                L_cup_S_fmat = torch.stack([remain_id_feature_mapping[id] for id in L_cup_S]).T.to(CONF.DEVICE)
                n = len(U_diff_S)
                max_info = -sys.maxsize
                sel_ind = sys.maxsize
                for i in range(0, n, GAP):
                    start = i
                    end = min(i+GAP, n)
                    U_diff_S_fmat = torch.stack([remain_id_feature_mapping[id] for id in U_diff_S[start:end]])
                    tt_mat = torch.matmul(U_diff_S_fmat, L_cup_S_fmat)
                    II = -torch.max(tt_mat, axis=1)[0]
                    can_ind = torch.argmax(II).item()
                    if II[can_ind].item() > max_info:
                        max_info = II[can_ind].item()
                        sel_ind = can_ind + start
                torch.cuda.empty_cache()
                S_set.add(U_diff_S[int(sel_ind)])
                print("\r {}/{}".format(count, q_budget - exploitation_m), end="", flush=True)
            print()
        return np.asarray(list(S_set), dtype=np.int32)