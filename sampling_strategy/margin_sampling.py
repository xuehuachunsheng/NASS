from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from complexity1 import *

# Renormalization group multiscale complexity
class MarginStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"Entropy Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities and margin values")
        margins = {}
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for count, i in enumerate(self.unlabeled_idx):
                x, y = self.train_ds[i]
                x = x[None, ...].to(CONF.DEVICE)
                z = self.model(x)
                logits = self.classifier(z)
                probs = torch.nn.functional.softmax(torch.squeeze(logits))
                probs = probs.cpu().numpy()
                probs = np.sort(probs)
                margin = probs[-1] - probs[-2]
                margins[i] = margin
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        CONF.logger.info("Step 3. Sort samples by margin values and return respective idx with top-k minimum values.")
        margins = list(margins.items())
        # Ascending order
        margins = sorted(margins, key=lambda x: x[1])
        top_k_idx = [x[0] for x in margins[:self.budget]]
        selected_idx = np.asarray(top_k_idx, dtype=int)
        return selected_idx