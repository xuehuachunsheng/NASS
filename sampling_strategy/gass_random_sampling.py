from sampling_strategy.strategy_abc import BatchSampleSelectionStratgy
import numpy as np
import torch

from utils import pred_probs, stratify, true_sample_idx_each_class,compute_bc,compute_bc_knownpopulation
class GASSRandomSelectionStrategy(BatchSampleSelectionStratgy):
    def __init__(self, train_ds, labeled_idx, budget, **kwargs):
        super().__init__(train_ds, labeled_idx, budget, **kwargs)
        self.round = kwargs["round"]
    
    def select(self) -> np.ndarray:
        CONF = self.CONF
        CONF.logger.info(f"GASS+Random Selecting samples. ")
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            unlabeled_probs = {}
            for count, i in enumerate(self.unlabeled_idx):
                x, y = self.train_ds[i]
                x = x[None, ...].to(self.CONF.DEVICE)
                z = self.model(x)
                logits = self.classifier(z)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                unlabeled_probs[i] = probs.tolist()[0]
                
                if count % 1000 == 0:
                    CONF.logger.info(f"{count}/{len(self.unlabeled_idx)}")

        CONF.logger.info("Step 3. Stratify.")
        n_cs = true_sample_idx_each_class(trainset=self.train_ds, _labeled_idx=self.labeled_idx,CONF=CONF)
        n_cs = [len(x) for x in n_cs]
        pred_C_idx = stratify(unlabeled_probs,CONF)
        CONF.logger.info("The sample size of each class (Model Predicted): {}".format(str([len(x) for x in pred_C_idx])))
        b_cs = compute_bc(n_cs,CONF)
        CONF.logger.info("Optimal query sample size: bc: {}".format(str(b_cs)))
        # 各个类别内随机抽样，前提是保持类别平衡
        # 需考虑样本不足的问题
        select_idx = [0] * CONF.n_classes
        adequates = [True] * CONF.n_classes
        for i in range(CONF.n_classes):
            pred_c_idx = pred_C_idx[i]
            bc = b_cs[i]
            select_c_idx = list(np.random.choice(pred_c_idx, size=min(len(pred_c_idx), bc), replace=False))
            adequates[i] = len(pred_c_idx) >= bc
            select_idx[i] = select_c_idx
        budgeted = np.sum([len(x) for x in select_idx])
        CONF.logger.info("Query Budget: {}, Satisfied：{}".format(self.budget, budgeted))
        assert budgeted <= self.budget
        class_id = -1
        if budgeted < self.budget:
            CONF.logger.warning("Polling Supply.")
        while budgeted < self.budget: # 采用轮询的方法抽取剩余样本    
            class_id = (class_id + 1) % CONF.n_classes
            if adequates[class_id]:
                rem_idx = set(pred_C_idx[class_id]).difference(set(select_idx[class_id]))
                rem_idx = list(rem_idx)
                if len(rem_idx) == 0:
                    adequates[class_id] = False
                    continue
                id = np.random.choice(rem_idx)
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