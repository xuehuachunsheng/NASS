from abc import ABC, abstractmethod

import os,sys
import torch
import numpy as np
from PIL import Image

class BatchSampleSelectionStratgy(ABC):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__()
        self.train_ds = train_ds
        self.labeled_idx = labeled_idx
        self.budget = budget
        all_idx = [i for i in range(len(train_ds))]
        self.unlabeled_idx = np.asarray(list(set(all_idx).difference(labeled_idx)), dtype=int)
        self.model = kwargs.get("model", None)
        self.classifier = kwargs.get("classifier", None)
        self.CONF = kwargs["CONF"] 
        

    @abstractmethod
    def select(self) -> np.ndarray:
        pass

    def __str__(self):
        s = self.__class__.__name__ + "\n"
        s += "Dataset size: {}\n".format(len(self.train_ds))
        s += "Labeled size: {}\n".format(len(self.labeled_idx))
        s += "Unlabeled size: {}\n".format(len(self.unlabeled_idx))
        s += "Budget: {}\n".format(self.budget)
        return s
