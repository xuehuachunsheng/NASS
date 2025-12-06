
import os,sys,logging
from typing import Any
import time
import torch
from losses import *

class Config_NASS:
    '''
    info_metric: choice from ["random", "entropy", "margin", "leastconfidence"]
    '''
    def __init__(self, **kwargs):
        # 邻域半径
        self.gamma = kwargs["gamma"]
        # 似然权重
        self._lambda = kwargs["_lambda"]
        
    def __str__(self):
        s = "Current Parameter Config: \n"
        members = vars(self).items()
        for k, v in members:
            s += f"{k}: {str(v)} \n"
        return s
