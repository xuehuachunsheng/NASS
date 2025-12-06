from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch
from complexity1 import *
from utils import sup_loss

# Renormalization group multiscale complexity
class ExpectedModelChangeStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]
        self.optimizer = kwargs["optimizer"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        self.model.eval()
        self.classifier.eval()
        pred_grads = {}
        for count, i in enumerate(self.unlabeled_idx):
            x, y = self.train_ds[i]
            x = x[None, ...].to(CONF.DEVICE)
            y = y[None, ...].to(CONF.DEVICE)
            z = self.model(x)
            logits = self.classifier(z)
            loss = sup_loss(logits, y)
            loss.backward()
            c_grad = 0
            for name,param in self.model.named_parameters():
                c_grad += torch.abs(param.grad).sum().item() if param.grad is not None else 0
            self.optimizer.zero_grad() # 清除梯度
            pred_grads[i] = c_grad
            if count % 1000 == 0:
                CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        CONF.logger.info("Step 3. Sort samples by emc values and return respective idx with top-k values.")
        pred_grads = list(pred_grads.items())
        pred_grads = sorted(pred_grads, key=lambda x: -x[1])
        top_k_idx = [x[0] for x in pred_grads[:self.budget]]
        selected_idx = np.asarray(top_k_idx, dtype=int)
        return selected_idx