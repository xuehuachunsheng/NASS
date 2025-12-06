from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from complexity1 import *

# Renormalization group multiscale complexity
class VarianceReductionStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"Entropy Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities and variances")
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            vars = {}
            for count, i in enumerate(self.unlabeled_idx):
                x, y = self.train_ds[i]
                x = x[None, ...].to(CONF.DEVICE)
                z = self.model(x)
                logits = self.classifier(z)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs = torch.squeeze(probs).cpu().numpy()
                #ent = -np.sum((probs + 1e-6) * np.log(probs + 1e-6))
                # 方差越小，越应该被加入到模型中，因为它能带来显著的期望方差降低
                var = np.var(probs)
                vars[i] = var
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        CONF.logger.info("Step 3. Sort samples by variances and return respective idx with top-k values.")
        vars = list(vars.items())
        vars = sorted(vars, key=lambda x: x[1])
        top_k_idx = [x[0] for x in vars[:self.budget]]
        selected_idx = np.asarray(top_k_idx, dtype=int)
        return selected_idx