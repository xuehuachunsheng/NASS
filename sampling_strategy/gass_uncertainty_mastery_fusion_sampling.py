import torch.utils
import torch.utils.data
from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from complexity1 import *

from utils import stratify,true_sample_idx_each_class
# Renormalization group multiscale complexity
# Refer to Wu et al. <Self-supervised class-balanced active learning with uncertainty-mastery fusion>
class GASSUncertaintyMasteryFusionStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"Entropy Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities and entropies")
        unlabeled_probs = {}
        unlabeled_entropies = {}
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for count, i in enumerate(self.unlabeled_idx):
                x, y = self.train_ds[i]
                x = x[None, ...].to(CONF.DEVICE)
                z = self.model(x)
                logits = self.classifier(z)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs = torch.squeeze(probs).cpu().numpy()
                ent = -np.sum((probs + 1e-6) * np.log(probs + 1e-6))
                unlabeled_entropies[i] = ent
                unlabeled_probs[i] = probs
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        pred_C_idx = stratify(unlabeled_probs,CONF)
        remain_mastery = self.compute_mastery()
        for i in range(len(pred_C_idx)):
            pred_C_idx[i] = [(id, unlabeled_entropies[id] * remain_mastery[id]) for id in pred_C_idx[i]]
            pred_C_idx[i] = sorted(pred_C_idx[i], key=lambda x: -x[1])
            pred_C_idx[i] = [x[0] for x in pred_C_idx[i]]
        
        n_cs = true_sample_idx_each_class(trainset=self.train_ds, _labeled_idx=self.labeled_idx,CONF=CONF)
        n_cs = [len(x) for x in n_cs]
        b_cs = self.compute_bc(n_cs,CONF)
        CONF.logger.info("The sample size of each class (Model Predicted): {}".format(str([len(x) for x in pred_C_idx])))
        CONF.logger.info("Optimal query sample size: bc: {}".format(str(b_cs)))
        # 各个类别内按照信息量的大小抽样
        # 需考虑样本不足的问题
        C = CONF.n_classes
        select_idx = [0] * C
        adequates = [True] * C
        for i in range(C):
            pred_c_idx = pred_C_idx[i]
            bc = b_cs[i]
            # 选择前面的样本，因为其信息量更大
            select_c_idx = pred_c_idx[:min(len(pred_c_idx), bc)]
            adequates[i] = len(pred_c_idx) >= bc
            select_idx[i] = select_c_idx
        budgeted = np.sum([len(x) for x in select_idx])
        CONF.logger.info("Query Budget: {}, Satisfied：{}".format(self.budget, budgeted))
        assert budgeted <= self.budget
        if budgeted < self.budget:
            CONF.logger.warning("Polling Supply.")
        class_id = -1
        while budgeted < self.budget: # 采用轮询的方法抽取剩余样本以满足预算要求   
            class_id = (class_id + 1) % C
            if adequates[class_id]:
                if len(pred_C_idx[class_id]) == len(select_idx[class_id]):
                    adequates[class_id] = False
                    continue
                id = pred_C_idx[class_id][len(select_idx[class_id])]
                select_idx[class_id].append(id)
                budgeted += 1
        #imb_index = compute_imb_index([len(x) for x in select_idx], conf, True)
        #logger.info("平均不平衡指数：{:.4f}".format(imb_index))
        f = open(CONF.bc_file, "a")
        b_csstr = ",".join([str(len(x)) for x in select_idx])
        f.write(str(self.round) + "," + b_csstr + "\n")
        f.close()
        sampling_idx = np.concatenate(select_idx)
        return sampling_idx
    
    def compute_mastery(self):
        CONF = self.CONF
        model = self.model
        remain_idx = self.unlabeled_idx
        dataset = self.train_ds
        model.eval()
        dc = 0.05 # cutoff distance ratio
        subset = torch.utils.data.Subset(dataset, remain_idx)
        n = len(subset)
        bs = CONF.bs
        c_loader = torch.utils.data.DataLoader(subset, batch_size=bs, shuffle=False, num_workers=0)
        # 存储所有样本的特征矩阵，特征的维度是512
        print("Compute the feature matrix of the unlabeled pool")
        fea_mats = np.empty((n, 512))
        with torch.no_grad():
            for i, (x_batch, _) in enumerate(c_loader):
                x_batch = x_batch.to(CONF.DEVICE)
                features = model.get_features(x_batch)
                features = features.cpu().numpy()
                fea_mats[i*bs:min((i+1)*bs, n)] = features
        f_mat = torch.tensor(fea_mats).to(CONF.DEVICE)
        del fea_mats
        # According to GPU memeroy size
        # 如果有多任务，则需降低GAP
        GAP = 100
        # Step 1. 计算真实的cutoff distance dc
        print("Computed the cutoff distance dc")
        max_dist = 0
        with torch.no_grad():
            for i in range(0,n,GAP): 
                start = i
                end = i + GAP if i + GAP < n else n
                part_distmat = torch.cdist(f_mat[start:end], f_mat) 
                c_max_dist = torch.max(part_distmat).data.item()
                if c_max_dist > max_dist:
                    max_dist = c_max_dist
                del part_distmat
                #print("\r{}/{}".format(i,n), end="", flush=True)
        real_dc = dc * max_dist
        print("d_c ratio: {}, d_c: {}".format(dc, real_dc))
        torch.cuda.empty_cache()
        # Step 2. 计算rho
        print("Compute Rho....")
        rho = np.zeros(n)
        with torch.no_grad():
            for i in range(0,n,GAP): 
                start = i
                end = i + GAP if i + GAP < n else n
                part_distmat = torch.cdist(f_mat[start:end], f_mat)
                for j in range(start, end):
                    dist_vec = part_distmat[j-start]
                    rho[j] = torch.sum(torch.where(dist_vec<real_dc,1,0)).data.item()
                    if (j+1) % 2000 == 0:
                        print("\rRho: {}/{}".format(j+1, n),end="",flush=True) 
                del part_distmat
                torch.cuda.empty_cache()
        print("\nAverage Rho: ", np.mean(rho))
        print("Max Rho: ", np.max(rho))
        print("Min Rho: ", np.min(rho))
        # Step 2. 计算delta            
        print("Compute Delta....")
        delta = np.zeros(n)
        with torch.no_grad():
            for i in range(0,n,GAP): 
                start = i
                end = min(i + GAP, n)
                part_distmat = torch.cdist(f_mat[start:end], f_mat)
                for j in range(start, end):
                    # 样本xj与其他所有样本的距离
                    xj_dist_vec = part_distmat[j-start]
                    # 满足gamma条件的最小距离
                    sat_idx = list(np.where(rho > rho[j])[0])
                    delta[j] = torch.min(xj_dist_vec[sat_idx]).item() if len(sat_idx) != 0 else max_dist
                    if (j+1)%2000 == 0:
                        print("\rDelta: {}/{}".format(j+1, n),end="",flush=True)    
                del part_distmat
                torch.cuda.empty_cache()
            print()
        f_mat = None # 释放显存空间
        mastery = {}
        for i in range(n):
            mastery[remain_idx[i]] = rho[i] * delta[i]
        return mastery