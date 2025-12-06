from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch
from complexity1 import *

from utils import stratify,compute_bc,true_sample_idx_each_class
# Renormalization group multiscale complexity
class GASSLeastConfidenceStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.model = kwargs["model"]
        self.classifier = kwargs["classifier"]
        self.round = kwargs["round"]

    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"Entropy Selecting samples. ")
        CONF.logger.info("Step 1. Compute the classification probabilities and entropies")
        unlabeled_probs = {}
        unlabeled_leastconfidences = {}
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
                unlabeled_probs[i] = probs
                lc = np.max(probs)
                unlabeled_leastconfidences[i] = lc
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")
        
        CONF.logger.info("Step 3. Stratify.")
        pred_C_idx = stratify(unlabeled_probs,CONF)
        for i in range(len(pred_C_idx)):
            # 按照MSC值大小进行分组并排序
            pred_C_idx[i] = [(id, unlabeled_leastconfidences[id]) for id in pred_C_idx[i]]
            # 按照值从小到大排序
            pred_C_idx[i] = sorted(pred_C_idx[i], key=lambda x: x[1])
            pred_C_idx[i] = [x[0] for x in pred_C_idx[i]]
        
        n_cs = true_sample_idx_each_class(trainset=self.train_ds, _labeled_idx=self.labeled_idx,CONF=CONF)
        n_cs = [len(x) for x in n_cs]
        b_cs = compute_bc(n_cs,CONF)
        CONF.logger.info("The sample size of each class (Model Predicted): {}".format(str([len(x) for x in pred_C_idx])))
        CONF.logger.info("Optimal query sample size: bc: {}".format(str(b_cs)))
        # 各个类别内按照信息量的大小抽样
        # 需考虑样本不足的问题
        select_idx = [0] * CONF.n_classes
        adequates = [True] * CONF.n_classes
        for i in range(CONF.n_classes):
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
            class_id = (class_id + 1) % CONF.n_classes
            if adequates[class_id]:
                if len(pred_C_idx[class_id]) == len(select_idx[class_id]):
                    adequates[class_id] = False
                    continue
                id = pred_C_idx[class_id][len(select_idx[class_id])]
                select_idx[class_id].append(id)
                budgeted += 1
        #imb_index = compute_imb_index([len(x) for x in select_idx], conf, True)
        #logger.info("平均不平衡指数：{:.4f}".format(imb_index))
        f = open(CONF.bc_file, "a")
        b_csstr = ",".join([str(len(x)) for x in select_idx])
        f.write(str(self.round) + "," + b_csstr + "\n")
        f.close()
        sampling_idx = np.concatenate(select_idx)
        return sampling_idx