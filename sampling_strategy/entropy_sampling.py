from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

class EntropyStrategy(BatchSampleSelectionStratgy):
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
            entropies = {}
            for count, i in enumerate(self.unlabeled_idx):
                x, y = self.train_ds[i]
                x = x[None, ...].to(CONF.DEVICE)
                z = self.model(x)
                logits = self.classifier(z)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs = torch.squeeze(probs).cpu().numpy()
                ent = -np.sum((probs + 1e-6) * np.log(probs + 1e-6))
                entropies[i] = ent
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        # 按照entropy值从大到小排序
        CONF.logger.info("Step 3. Sort samples by entropy values and return respective idx with top-k values.")
        entropies = list(entropies.items())
        entropies = sorted(entropies, key=lambda x: -x[1])
        top_k_idx = [x[0] for x in entropies[:self.budget]]
        selected_idx = np.asarray(top_k_idx, dtype=int)
        return selected_idx