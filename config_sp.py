import os,sys
from datetime import datetime,timedelta
import pandas as pd

class Config_SP:

    def __init__(self, **kwargs):

        # 是否是预警，如果是预警，则只把卡钻发生前的征兆当作卡钻标签
        self.early_warning = True
        
        self.seed = 3407

        # 如果是二分类，则只区分正常钻进和卡钻
        self.binary = False

        self.sp_classnames = ['压差卡钻', '坍塌卡钻', '沉砂卡钻', '缩径卡钻', '落物卡钻', '键槽卡钻', '正常钻进']
        self.sp_classnames_english = ['Differential Sticking', 
                                      'Collapse Sticking', 
                                      'Sand Bridging', # 沉砂，也叫砂桥
                                      'Reducing Clamp',  
                                      'Junk Sticking', 
                                      'Keyseat Sticking', 
                                      'Normal Working']
        # 根据专家经验获取的每个类别在总体中的比例
        #self.population_ratio = [0.035, 0.02, 0.02, 0.005, 0.005, 0.015, 0.9]
        self.population_ratio = [0.175, 0.1, 0.1, 0.025, 0.025, 0.075, 0.5]
        self.exclude_class_idx = [6] # 在计算精确率等指标时，是否排除正常钻进类别
        self.n_classes = len(self.sp_classnames)
        self.original_features = [
            "日期","时间","井深","层位","钻头位置",
            "迟到位置","迟到时间","大钩负荷","钻压","钻时",
            "大钩高度","转盘转速","扭矩","钻时","套压",
            "立压","泵冲1","泵冲2","泵冲3","总泵冲",
            "钻井状态","入口流量","出口流量","入口密度","出口密度",
            "入口温度","出口温度","入口电导","出口电导","溢漏",
            "总池体积","总烃","二氧化碳","C1","C2",
            "C3","iC4","nC4","iC5","nC5",
            "硫化氢1","硫化氢2","硫化氢3","硫化氢4","池体积1",
            "池体积2","池体积3","池体积4","池体积5","池体积6",
            "池体积7","池体积8","池体积9","池体积10","池体积11"
        ]
        self.features = [
            "日期","时间","井深","钻头位置","迟到位置",
            "迟到时间","大钩负荷","钻压","钻时","大钩高度",
            "转盘转速","扭矩","套压","立压","总泵冲",
            "钻井状态","入口流量","出口流量","入口密度","出口密度",
            "入口温度","出口温度","入口电导","出口电导","溢漏", 
            "总池体积","总烃"
        ]
        self.numeric_features = [     "井深","钻头位置","迟到位置",
                                    "迟到时间","大钩负荷","钻压","钻时","大钩高度",
                                    "转盘转速","扭矩","套压","立压","总泵冲", 
                                    "入口流量","出口流量","入口密度","出口密度",
                                    "入口温度","出口温度","入口电导","出口电导","溢漏", 
                                    "总池体积","总烃"] # 所有数值型特征
        # 所有层位
        #self.all_layers = ['东岳庙段', '凉上段', '吴家坪组', '嘉一段', '嘉三段', '嘉二3亚段', '嘉四段', '大安寨段', '栖一段', '栖二段', '栖霞组', '梁山组', '沙一段', '沙二段', '沙溪庙组', '沧浪铺组', '灯四段', '筇竹寺组', ' 自流井组', '茅一段', '茅三段', '茅二c段', '茅二段', '茅口组', '蓬莱镇组', '费三段', '长兴组', '雷二段', '须三段', '须四段', '须家河组', '飞一段', '飞三段', '飞二段', '马鞍山段', '龙潭组', '龙王庙组', '龙马溪组']
        self.categorical_features = ['钻井状态']       # 所有标称型特征
        self.all_status = ['下钻', '划眼', '坐卡', '循环', '挂起', 
                           '接单根', '接立柱', '接触井底', '提离井底', '提钻', 
                           '未知', '等待', '起钻', '钻进']
        self.category_values = [self.all_status] # 每个标称特征的所有取值
        
        self.root_path = "E:\\Datasets\\钻采院卡钻数据\\数据"

        self.all_dirs = sorted(os.listdir(self.root_path))
        
        self.file_label_mapping = self.load_files()

        self.memory_size = 2590 # 所有数据占用的内存大小为2590MB，仅供参考

        # 原始数据里面采用了插值方法，生成了每秒的数据
        self.seq_len = kwargs.get("T", 180) # 一条数据的长度为seq_len，可以设置为60,120,180
        self.early_warning_len = 3600*4 # 前驱信号序列长度，提前预警时长为 early_warning_len*step s
        
        # 一条数据里面每个数据元素的时间间隔，为1表示间隔1s，相邻序列为1s步长，为2表示间隔2s，相邻序列为2s步长
        # 一条数据的时间长度是seq_len*step s
        self.step = 1 
        self.n_features = 38

        hours = (self.early_warning_len * self.step) //3600
        minutes = (self.early_warning_len * self.step % 3600) // 60
        seconds = (self.early_warning_len * self.step) % 60
        self.early_warning_time = timedelta(hours=hours,minutes=minutes,seconds=seconds)

        # 训练集验证集比例
        self.train_ratio,self.test_ratio = 0.95,0.05

    def load_files(self):
        data_file_paths = {}
        for dir in self.all_dirs:
            # 处理正常数据
            if dir in ["东坝1-正常井数据","长宁H20-1-正常井数据"]:
                data_file_paths[os.path.join(self.root_path, dir, "卡钻前-正常数据-1s-norm.csv")] = self.sp_classnames.index("正常钻进")
                continue
            # 处理卡钻数据
            sp_type = dir.split("-")[-1]
            if not dir.endswith("卡钻"):
                sp_type = dir.split("-")[-2]
            assert sp_type in self.sp_classnames
            data_file_paths[os.path.join(self.root_path, dir, "卡钻前-正常数据-1s-norm.csv")] = self.sp_classnames.index("正常钻进")
            #data_file_paths[os.path.join(self.root_path, dir, "卡钻后-正常数据-1s-norm.csv")] = self.sp_classnames.index("正常钻进")
            if not self.early_warning:
                data_file_paths[os.path.join(self.root_path, dir, "卡钻前-异常数据-1s-norm.csv")] = self.sp_classnames.index("正常钻进")
                data_file_paths[os.path.join(self.root_path, dir, "卡钻中-异常数据-1s-norm.csv")] = self.sp_classnames.index(sp_type)
            else:
                data_file_paths[os.path.join(self.root_path, dir, "卡钻前-异常数据-1s-norm.csv")] = self.sp_classnames.index(sp_type)

                
        # 筛选出没有实质数据的文件
        no_data_file_paths = []
        for k in data_file_paths:
            df = pd.read_csv(k,encoding="gbk")
            if df.size == 0:
                no_data_file_paths.append(k)
        for k in no_data_file_paths:
            data_file_paths.pop(k)
        return data_file_paths
    
    def __str__(self):
        s = "Current Parameter Config for SP: \n"
        members = vars(self).items()
        for k, v in members:
            s += f"{k}: {v} \n"
        return s

#CONFSP = Config_SP()

if __name__ == "__main__":
    import os,sys
    import pandas as pd
    import numpy as np
    conf = Config_SP()
    print([x[1] for x in conf.file_label_mapping.items()])
    