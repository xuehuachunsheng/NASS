from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from complexity1 import *

# Renormalization group multiscale complexity
class LeastConfidenceStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"Entropy Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities and entropies")
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            least_confidences = {}
            for count, i in enumerate(self.unlabeled_idx):
                x, y = self.train_ds[i]
                x = x[None, ...].to(CONF.DEVICE)
                z = self.model(x)
                logits = self.classifier(z)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs = torch.squeeze(probs).cpu().numpy()
                lc = np.max(probs)
                least_confidences[i] = lc
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        # 按照entropy值从大到小排序
        CONF.logger.info("Step 3. Sort samples by least confidence values and return respective idx with top-k minimum values.")
        least_confidences = list(least_confidences.items())
        # ascending order
        least_confidences = sorted(least_confidences, key=lambda x: x[1])
        top_k_idx = [x[0] for x in least_confidences[:self.budget]]
        selected_idx = np.asarray(top_k_idx, dtype=int)
        return selected_idx