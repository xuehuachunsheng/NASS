# Entropy with local empirical bayes correction with global prior 
# Do not use stratify

from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from utils import find_all_neighbors_in_X

class LEBGPMarginStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.gamma = kwargs["gamma"] # 邻域半径 相对于边际距离的比例
        self._lambda = kwargs["_lambda"] # 似然权重

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info("Margin using Local Empirical Bayes with Global Prior Correction.")
        CONF.logger.info("Step 1. Compute model probabilities and features of all samples")
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
        CONF.logger.info("Step 2.2. Compute LEBGP Margin")
        PY = np.array(CONF.CONFSP.population_ratio)
        probs_np = probs.detach().cpu().numpy()
        all_features_np = all_features.detach().cpu().numpy()
        assert len(all_features_np.shape) == len(probs_np.shape) == 2
        Xneighbors = find_all_neighbors_in_X(all_features_np, cutoff_dist, True)
        maximum_neighbor_num = np.max([len(x) for x in Xneighbors])
        tildePN = {}
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

        # 计算Margin
        lebgp_margins = {}
        for k in tildePN:
            sortedPN = np.sort(tildePN[k])
            lebgp_margins[k] = sortedPN[-1] - sortedPN[-2]

        # 按照margin值从小到大排序
        CONF.logger.info("Step 3. Sort samples by LEBGP margin values and return respective idx with top-k values.")
        lebgp_margins = list(lebgp_margins.items())
        lebgp_margins = sorted(lebgp_margins, key=lambda x: x[1])
        top_k_idx = [x[0] for x in lebgp_margins[:self.budget]]
        selected_idx = np.asarray(top_k_idx, dtype=int)
        return selected_idx