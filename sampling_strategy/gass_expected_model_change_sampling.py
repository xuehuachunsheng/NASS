from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch
from complexity1 import *

from utils import stratify,true_sample_idx_each_class,compute_bc,sup_loss
# Renormalization group multiscale complexity
class GASSExpectedModelChangeStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]
        self.optimizer = kwargs["optimizer"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"Entropy Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities and entropies")
        self.model.eval()
        self.classifier.eval()
        pred_grads = {}
        unlabeled_probs = {}
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
            pred_grads[i] = c_grad
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = torch.squeeze(probs).detach().cpu().numpy()
            unlabeled_probs[i] = probs
            self.optimizer.zero_grad() # 清除梯度

            if count % 1000 == 0:
                CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        # 按照entropy值从大到小排序
        CONF.logger.info("Step 3. Sort samples by emc values and return respective idx with top-k values.")
        pred_C_idx = stratify(unlabeled_probs,CONF)
        for i in range(len(pred_C_idx)):
            pred_C_idx[i] = [(id, pred_grads[id]) for id in pred_C_idx[i]]
            pred_C_idx[i] = sorted(pred_C_idx[i], key=lambda x: -x[1])
            pred_C_idx[i] = [x[0] for x in pred_C_idx[i]]
        
        n_cs = true_sample_idx_each_class(trainset=self.train_ds, _labeled_idx=self.labeled_idx,CONF=CONF)
        n_cs = [len(x) for x in n_cs]
        b_cs = compute_bc(n_cs,CONF)
        CONF.logger.info("The sample size of each class (Model Predicted): {}".format(str([len(x) for x in pred_C_idx])))
        CONF.logger.info("Optimal query sample size: bc: {}".format(str(b_cs)))
        # 各个类别内按照信息量的大小抽样
        # 需考虑样本不足的问题
        C = CONF.n_classes
        select_idx = [0] * C
        adequates = [True] * C
        for i in range(C):
            pred_c_idx = pred_C_idx[i]
            bc = b_cs[i]
            # 选择前面的样本，因为其信息量更大
            select_c_idx = pred_c_idx[:min(len(pred_c_idx), bc)]
            adequates[i] = len(pred_c_idx) >= bc
            select_idx[i] = select_c_idx
        budgeted = np.sum([len(x) for x in select_idx])
        CONF.logger.info("Query Budget: {}, Satisfied：{}".format(self.budget, budgeted))
        assert budgeted <= self.budget
        if budgeted < self.budget:
            CONF.logger.warning("Polling Supply.")
        class_id = -1
        while budgeted < self.budget: # 采用轮询的方法抽取剩余样本以满足预算要求   
            class_id = (class_id + 1) % C
            if adequates[class_id]:
                if len(pred_C_idx[class_id]) == len(select_idx[class_id]):
                    adequates[class_id] = False
                    continue
                id = pred_C_idx[class_id][len(select_idx[class_id])]
                select_idx[class_id].append(id)
                budgeted += 1
        #imb_index = compute_imb_index([len(x) for x in select_idx], conf, True)
        #logger.info("平均不平衡指数：{:.4f}".format(imb_index))
        sampling_idx = np.concatenate(select_idx)
        return sampling_idx