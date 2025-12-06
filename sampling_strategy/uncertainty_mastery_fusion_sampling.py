import torch.utils
import torch.utils.data
from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from complexity1 import *

# Renormalization group multiscale complexity
# Refer to Wu et al. <Self-supervised class-balanced active learning with uncertainty-mastery fusion>
class UncertaintyMasteryFusionStrategy(BatchSampleSelectionStratgy):
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
        
        remain_mastery = self.compute_mastery()
        remain_ent = unlabeled_entropies.items()
        remain_ent = sorted(remain_ent, key=lambda x: x[0])
        remain_mastery = remain_mastery.items()
        remain_mastery = sorted(remain_mastery, key=lambda x: x[0])
        idx = [x[0] for x in remain_ent]
        assert idx == [x[0] for x in remain_mastery]
        remain_infos = list(np.asarray([x[1] for x in remain_ent]) * np.asarray([x[1] for x in remain_mastery]))
        remain_infos = list(zip(idx, remain_infos))
        remain_infos = sorted(remain_infos, key=lambda x: -x[1])
        sampling_idx = np.asarray([x[0] for x in remain_infos[:CONF.B]], dtype=np.int32)
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