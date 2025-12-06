
import os,sys,logging
from typing import Any
import time
import torch
from losses import *

class _ParamConfig:
    def __init__(self, **kwargs):
        self.CONFSP = kwargs["CONFSP"]
        self.CONFNASS = kwargs["CONFNASS"]
        self.data = "sp" # 卡钻数据
        self.n_classes = 7
        
        self.E1 = 21 # Query epochs
        self.E2 = 30 # Training epochs
        self.init_labels = 7

        self.seed = 3407
        self.use_cuda = True
        self.gass = kwargs.get("gass", False) 
        self.nass = kwargs.get("nass", False)
        assert not (self.nass and self.gass)  # gass和nass只能有一个为真，或者都为假
        self.known_population = True # If the population is know 
        self.population_ratio = None
        if self.known_population:
            self.population_ratio = self.CONFSP.population_ratio
            
        self.a2s2 = self.gass # 没有用到这个参数
        self.method = kwargs.get("method","random")
        self.B = kwargs.get("B", 50)  # The budget of each AL round can be 30, 50
        self.lr = 1e-3
        self.bs = 128
        self.dynamic_E2 = True

        local_time = time.localtime()
        local_time = time.strftime("%Y%m%d-%H_%M_%S", local_time)
        local_dir = f"models/{self.data}/{local_time}/"
        os.makedirs(local_dir)
        self.model_file = local_dir + "best_model.pth"
        self.train_loss_file = local_dir + "train_loss.csv"
        self.val_loss_file = local_dir + "val_loss.csv"
        self.val_acc_file = local_dir + "val_acc.csv"
        self.nc_file = local_dir + "nc.csv"
        self.bc_file = local_dir + "bc.csv"
        self.conf_file = local_dir + "conf.txt"
        self.js_file = local_dir + "js_div.csv"

        f = open(self.train_loss_file, "w")
        f.write("Round,Epoch,SUP_LOSS\n")
        f.close()
        f = open(self.val_loss_file, "w") 
        f.write("Round,SUP_LOSS\n")
        f.close()
        f = open(self.val_acc_file, "w")
        f.write("Round,WP,WR,WF\n")
        f.close()
        t = ",".join([f"c_{i}" for i in range(self.n_classes)])
        f = open(self.nc_file, "w")
        f.write("Round,{}\n".format(t))
        f.close()
        f = open(self.bc_file, "w")
        f.write("Round,{}\n".format(t))
        f.close()
        f = open(self.js_file, "w")
        f.write("Round,JS\n")
        f.close()

        # 取值范围限定
        assert self.data == "sp"
        assert self.gass in [True, False]
        assert self.method in ["random", "entropy", "leastconfidence","margin", "vr", "emc",
                               "lebgp-entropy", "lebgp-leastconfidence", "lebgp-margin"]
        #assert self.train_mode in ["full", "sup"]

        logger = logging.getLogger('Log: ')  
        logger.setLevel(logging.INFO)
        # STDOUT
        self.ch = logging.StreamHandler(stream=sys.stdout)  
        fmt = logging.Formatter("%(asctime)s | %(message)s")
        self.ch.setLevel(logging.INFO)
        self.ch.setFormatter(fmt)
        # FILE_OUT
        self.log_file = local_dir + "out.log"
        self.fh = logging.FileHandler(self.log_file,encoding="utf8",mode='w')
        self.fh.setLevel(logging.INFO) 
        logger.addHandler(self.ch)
        logger.addHandler(self.fh)
        self.logger = logger
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        self.loss_name = kwargs.get("loss", "ce")

        if self.loss_name == "twl":
            weight = np.asarray(self.CONFSP.population_ratio) * self.B * (self.E1 - 1)
            weight = weight / np.sum(weight)*len(weight)
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).float(), reduction='mean')
        elif self.loss_name == 'ce':
            self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        elif self.loss_name == 'fl':
            self.loss = FocalLoss(gamma=2, reduction="mean")
        elif self.loss_name == 'eql':
            freq_ratio = np.asarray(self.CONFSP.population_ratio)
            self.loss = EQLloss(freq_ratio)
        elif self.loss_name == 'cbl':
            samples_per_class = np.asarray(self.CONFSP.population_ratio) * self.B * (self.E1 - 1)
            self.loss = CB_Softmax(samples_per_class,reduction="mean")
        elif self.loss_name == 'cdbl':
            self.loss = CDB_loss(class_difficulty = np.ones(self.n_classes), tau="dynamic",reduction="mean")
        elif self.loss_name == 'dqbl':
            self.loss = DQB_loss(C=self.n_classes, k=3, alpha="dynamic",reduction="mean")
        else:
            sys.exit('Sorry. No such loss function implemented')

        self.loss = self.loss.to(self.DEVICE)

        f = open(self.conf_file, "w", encoding="utf-8")
        #f.write(str(self))
        f.write(f"using nass:{self.nass}\n")
        f.write(f"info metric:{self.method}\n")
        f.write(f"B:{self.B}\n")
        f.write(f"gamma:{self.CONFNASS.gamma} (gamma=0 indicates using global prior only without LEB)\n")
        f.write(f"lambda:{self.CONFNASS._lambda} (lambda=infty indicates using global prior only without LEB)\n")
        f.write("\n\n\nOther settings are as follows:\n")
        f.write(str(self))
        f.close()

        
    def __str__(self):
        s = "Current Parameter Config: \n"
        members = vars(self).items()
        for k, v in members:
            s += f"{k}: {str(v)} \n"
        return s

#CONF = _ParamConfig()