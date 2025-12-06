from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from complexity1 import *

# Renormalization group multiscale complexity
class RGMSCSelectionStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.feature_maps = kwargs["feature_maps"]
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"RGMSC Selecting samples. ")
        CONF.logger.info("Step 1. Compute the feature maps and rgmsc values.")
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            CONF.logger.info("Step 2. Computer rgmsc for all unlabeled samples.")
            msc_value_idx = {}
            for count, i in enumerate(self.unlabeled_idx):
                x, y = self.train_ds[i]
                x = x[None, ...].to(CONF.DEVICE)
                self.model(x)
                msc_value = 0
                for name, value in self.feature_maps.items():
                    value = value[0]
                    if value.shape[1] <= 4: 
                        continue
                    value = (value - torch.amin(value, dim=(-2,-1),keepdim=True)) / (torch.amax(value, dim=(-2,-1),keepdim=True) - torch.amin(value, dim=(-2,-1),keepdim=True) + 1e-6)
                    value = 2 * value - 1
                    kMax = int(np.log2(value.shape[1])) - 2  # For the full range maximal value
                    c_value = structuralComplexity_torch(value, kLargerThan=0, kMax=kMax)
                    msc_value += c_value.item()
                
                msc_value_idx[i] = msc_value
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        # 按照MSC值从大到小排序
        CONF.logger.info("Step 3. Sort samples by rgmsc values and return respective idx with top-k values.")
        msc_value_idx = list(msc_value_idx.items())
        for i, (id, value) in enumerate(msc_value_idx):
            if np.isnan(msc_value_idx[i][1]):
                msc_value_idx[i][1] = 0
        msc_value_idx = sorted(msc_value_idx, key=lambda x: -x[1])
        top_k_idx = [x[0] for x in msc_value_idx[:self.budget]]
        selected_idx = np.asarray(top_k_idx, dtype=int)
        return selected_idx