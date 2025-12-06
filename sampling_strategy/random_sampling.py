from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np

class RandomSelectionStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
    
    def select(self) -> np.ndarray:
        return np.random.choice(self.unlabeled_idx,size=self.budget, replace=False)