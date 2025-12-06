import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sigmoid(x):
  return (1/(1+np.exp(-x)))

class CDB_loss(nn.Module):
    def __init__(self, class_difficulty, tau='dynamic', reduction='none'):
        super(CDB_loss, self).__init__()
        self.class_difficulty = class_difficulty
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/(1 - np.max(class_difficulty) + 0.01)
            tau = sigmoid(bias)
        else:
            tau = float(tau) 
        self.weights = self.class_difficulty ** tau
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weights), reduction=self.reduction).cuda()
        
    def forward(self, input, target):
        return self.loss(input, target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss


class CB_loss(nn.Module):
     def __init__(self, beta, samples_per_cls, reduction='none'):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.samples_per_cls = samples_per_cls
        self.reduction = reduction
        self.device = torch.device('cuda:0')

     def compute_weights(self):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(self.samples_per_cls)
        self.weights = torch.Tensor(weights)
   
     def forward(self, input, target):
         raise NotImplementedError
    
class CB_Softmax(CB_loss):
    def __init__(self, samples_per_cls, beta=0.9, reduction='none',device="cuda"):      
        super().__init__(beta, samples_per_cls, reduction)
        self.compute_weights()
        self.loss = nn.CrossEntropyLoss(weight=self.weights, reduction=self.reduction)
        if device=="cuda":
            self.loss = self.loss.cuda()

    def forward(self, input, target):
        return self.loss(input, target)
 
class CB_Focal(CB_loss):
    def __init__(self, samples_per_cls, beta=0.9, gamma=1, reduction='none',device="cuda"):
        super().__init__(beta, samples_per_cls, reduction)
        self.compute_weights()
        self.loss = FocalLoss(alpha=self.weights, gamma=gamma)
        if device=="cuda":
            self.loss = self.loss.cuda()
    
    def forward(self, input, target):
        return self.loss(input, target)


# Discriminant quality based loss
class DQB_loss(nn.Module):
    # C: number of classes
    # k: time window size
    def __init__(self, C, k=3, alpha="dynamic", reduction="none", eps0=1e-3, eps1=1e-3, alphaM=2,device="cuda"):
        super(DQB_loss, self).__init__()
        self.C = C
        self.alpha = alpha
        self.k = k
        self.eps0,self.eps1,self.alphaM = eps0,eps1,alphaM
        self.H_M = np.tile(np.log(C), reps=(C,k))
        self.B = np.asarray([1 / (i + 1) for i in range(k)])
        self.B = self.B / np.sum(self.B)
        self.B = np.tile(self.B, reps=(C,1))
        self.A = np.zeros((C, k)) # acc for each class
        self.H = np.zeros((C, k)) # entropy for each class
        self.DQ = np.ones(C)
        self.weights = torch.ones(C).float()
        self.reduction=reduction
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weights), reduction=reduction)
        self.device=device
        if device=="cuda":
            self.loss = self.loss.cuda()
        
    def update_DQBWeight(self, acc, entropy):
        # 利用验证数据更新A, H
        self.A = np.hstack([np.zeros((self.C, 1)), self.A[:, :-1]])
        self.H = np.hstack([np.zeros((self.C, 1)), self.H[:, :-1]])
        #acc, entropy = self.computeAccEntropy(model, val_datagen)
        self.A[:, 0] = acc
        self.H[:, 0] = entropy
        self.DQ = np.sum(self.B * self.A * (self.H_M - self.H), axis=1)
        t_alpha = None
        if self.alpha != "dynamic":
            t_alpha = int(self.alpha)
        else:
            t_alpha = (np.max(self.DQ) + self.eps1) / (np.min(self.DQ) + self.eps1) - 1
            t_alpha = self.alphaM*(2/(1+np.exp(-t_alpha)) - 1)
        self.weights = np.power(np.max(self.DQ) - self.DQ + self.eps0, t_alpha)
        self.weights = self.weights / np.sum(self.weights) * len(self.weights)
        self.weights = torch.tensor(self.weights).float()
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weights), reduction=self.reduction)
        if self.device=="cuda":
            self.loss = self.loss.cuda()
    
    def computeAccEntropy(self, model, classifier, val_dataloader):
        acc = np.zeros(self.C)
        entropy = np.zeros(self.C)
        val_samples = np.zeros(self.C) # The number of valid samples of each class
        model.eval()
        classifier.eval()
        for imgs, labels in val_dataloader:
            imgs = imgs.cuda()
            labels = labels.type(torch.cuda.LongTensor)
            out = None
            with torch.no_grad():
                z = model(imgs)
                logits = classifier(z)
                probs = F.softmax(logits, dim=1)
            _, pred = torch.max(probs, dim=1)
            
            for id in range(len(pred)): 
                if pred[id] == labels[id]:
                    acc[int(labels[id])] += 1
                    # Accumulate the entropy
                    prob_x = probs[id].cpu().numpy()
                    ent_x = -np.sum(prob_x * np.log(prob_x)) # Compute information entropy
                    entropy[int(labels[id])] += ent_x
                val_samples[int(labels[id])] += 1
        entropy /= (acc + 1e-6) # convert to average information entropy
        acc /= val_samples # convert to 0-1 acc for each class
        return acc, entropy
                
    def forward(self, input, target):
        return self.loss(input, target)

class EQLloss(nn.Module):
    def __init__(self, freq_info, device="cuda"):
        super(EQLloss, self).__init__()
        self.freq_info = freq_info
        # self.pred_class_logits = pred_class_logits
        # self.gt_classes = gt_classes
        self.lambda_ = 0.03
        self.gamma = 0.95
        self.device=device

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def forward(self, pred_class_logits, gt_classes,):
        self.pred_class_logits = pred_class_logits
        self.gt_classes = gt_classes
        self.n_i, self.n_c = self.pred_class_logits.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(self.pred_class_logits, self.gt_classes)
        if torch.rand(1).item() > self.gamma:
            coeff = torch.zeros(1)
        else:
            coeff = torch.ones(1)
        coeff = coeff.cuda()
        eql_w = 1 - (coeff * self.threshold_func() * (1 - target))

        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, target,
                                                      reduction="none")
        if self.device=="cuda":
            return torch.sum(cls_loss * eql_w).cuda() / self.n_i
        return torch.sum(cls_loss * eql_w) / self.n_i