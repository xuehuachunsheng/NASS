from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch
from complexity1 import *

from utils import stratify,compute_bc,true_sample_idx_each_class,find_all_neighbors_in_X
# Renormalization group multiscale complexity
class NASSLEBGPLeastConfidenceStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.round = kwargs["round"]
        self.gamma = kwargs["gamma"] # 邻域半径 相对于边际距离的比例
        self._lambda = kwargs["_lambda"] # 似然权重

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"NASS LEBGP LeastConfidence Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities")
        self.model.eval()
        self.classifier.eval()
        n_samples = len(self.train_ds)
        probs = torch.empty((n_samples, CONF.n_classes)).to(CONF.DEVICE) # 也包括无标注样本的标签one-hot编码
        all_features = torch.empty((n_samples, 512)).to(CONF.DEVICE) # 也包括标注样本的logits
        labeled_idx_set = set(self.labeled_idx)
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=CONF.bs, shuffle=False)
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(CONF.DEVICE), y.to(CONF.DEVICE)
                z = self.model(x)
                all_features[int(i*CONF.bs):int(min((i+1)*CONF.bs, n_samples))] = z
                if i in labeled_idx_set:
                    prob = torch.nn.functional.one_hot(y, num_classes=CONF.n_classes) # 直接获取真实标签
                    prob = torch.tensor(prob, dtype=torch.float32)
                else:
                    logit = self.classifier(z)
                    prob = torch.nn.functional.softmax(logit, dim=1,dtype=torch.float32)
                probs[int(i*CONF.bs):int(min((i+1)*CONF.bs, n_samples))] = prob
                
                if i*CONF.bs % 100 == 0:
                    CONF.logger.info(f"{i*CONF.bs}/{len(self.unlabeled_idx)}")

        CONF.logger.info("Step 2. Compute LEBGP corrected probability")
        # 计算边际距离不能使用faiss加速，只能硬解，但可以使用分块方法
        CONF.logger.info("Step 2.1. Compute margin distance")
        max_dist = 0
        GAP = 500
        with torch.no_grad():
            for i in range(0,n_samples,GAP): 
                start = i
                end = i + GAP if i + GAP < n_samples else n_samples
                part_distmat = torch.cdist(all_features[start:end], all_features, p=2) 
                c_max_dist = torch.max(part_distmat).data.item()
                if c_max_dist > max_dist:
                    max_dist = c_max_dist
                del part_distmat
        cutoff_dist = self.gamma * max_dist

        # 计算邻居样本, 用faiss库加速近邻搜索
        CONF.logger.info("Step 2.2. Compute LEBGP Least Confidence")
        PY = np.array(CONF.CONFSP.population_ratio)
        probs_np = probs.detach().cpu().numpy()
        all_features_np = all_features.detach().cpu().numpy()
        assert len(all_features_np.shape) == len(probs_np.shape) == 2
        Xneighbors = find_all_neighbors_in_X(all_features_np, cutoff_dist, True)
        maximum_neighbor_num = np.max([len(x) for x in Xneighbors])
        tildePN = {}
        leastconfidences = {}
        for i in range(n_samples):
            if i in labeled_idx_set:
                continue
            xneighbor_idx = Xneighbors[i]
            Nx = len(xneighbor_idx)
            if self._lambda == "dym":
                _lambda = (maximum_neighbor_num - Nx + 1) / (Nx + 1)
            else:
                _lambda = self._lambda / (Nx + 1)
            fx = probs_np[i]
            alphaNx = np.sum(probs_np[xneighbor_idx], axis=0) / (Nx + 1)
            tildePNx = PY * (alphaNx + _lambda * fx) # 融入全局先验
            tildePNx = tildePNx / np.sum(tildePNx) # 归一化
            tildePN[i] = tildePNx
            leastconfidences[i] = np.max(tildePNx)
        
        unlabeled_probs = tildePN
        CONF.logger.info("Step 3. Stratify.")
        pred_C_idx = stratify(unlabeled_probs,CONF)
        for i in range(len(pred_C_idx)):
            # 按照MSC值大小进行分组并排序
            pred_C_idx[i] = [(id, leastconfidences[id]) for id in pred_C_idx[i]]
            pred_C_idx[i] = sorted(pred_C_idx[i], key=lambda x: x[1])
            pred_C_idx[i] = [x[0] for x in pred_C_idx[i]]
        
        n_cs = true_sample_idx_each_class(trainset=self.train_ds, _labeled_idx=self.labeled_idx,CONF=CONF)
        n_cs = [len(x) for x in n_cs]
        b_cs = compute_bc(n_cs,CONF)
        CONF.logger.info("The sample size of each class (Corrected Model Prediction): {}".format(str([len(x) for x in pred_C_idx])))
        CONF.logger.info("Optimal query sample size: bc: {}".format(str(b_cs)))
        # 各个类别内按照信息量的大小抽样
        # 需考虑样本不足的问题
        select_idx = [0] * CONF.n_classes
        adequates = [True] * CONF.n_classes
        for i in range(CONF.n_classes):
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
            class_id = (class_id + 1) % CONF.n_classes
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