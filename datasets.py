import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset,DataLoader,random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class MultivariateTSDataset(Dataset):
    """处理多个CSV文件的多元时间序列数据集，支持独热编码"""
    
    def __init__(self, CONFSP):
        # 初始化预处理管道
        self.CONFSP = CONFSP
        self._init_preprocessors()
        # 每个类别的样本数量
        # 预加载并预处理数据
        self._load_and_preprocess_data()
        
    def _init_preprocessors(self):
        
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(categories=self.CONFSP.category_values,sparse_output=False), self.CONFSP.categorical_features)],
            remainder="passthrough"
        )
        
        self.label_encoder = LabelEncoder()
    
    def _load_and_preprocess_data(self):
        """加载并预处理所有CSV文件"""
        all_sequences = []
        all_labels = []
        self.sizes_of_each_class = [0]*self.CONFSP.n_classes
        for i, file_path in enumerate(self.CONFSP.file_label_mapping):
            print(f"\r加载数据: {i}/{len(self.CONFSP.file_label_mapping)}",end="",flush=True)
            df = pd.read_csv(os.path.join(file_path),encoding="gbk")
            
            # 分离特征和标签
            features = df.drop(["日期", "时间"],axis=1)
            label = self.CONFSP.file_label_mapping[file_path]
            
            # 已经进行标准化了，因此只对状态进行独热处理
            #num_features = features[CONF.numeric_features]
            #cat_features = features[CONF.category_values]
            processed_features = self.preprocessor.fit_transform(features)
            
            # 分割为固定长度序列
            for i in range(0, df.shape[0] - self.CONFSP.seq_len*self.CONFSP.step + 1, self.CONFSP.seq_len*self.CONFSP.step):
                # 指定步长，也就是时间间隔
                seq_features = processed_features[i:i + self.CONFSP.seq_len*self.CONFSP.step:self.CONFSP.step]
                all_sequences.append(seq_features)
                all_labels.append(label)
                self.sizes_of_each_class[label] += 1
        print()

        # 转换为数组并编码标签
        self.features = np.stack(all_sequences,dtype=np.float32)
        # 转化为(n_samples, n_features, seq_len)
        self.features = np.transpose(self.features,(0,2,1))
        self.labels = np.asarray(all_labels, dtype=np.int32)
        assert self.features.shape[0] == self.labels.shape[0]
        # 获取特征维度
        self.feature_dim = self.features.shape[-1]
        
        # 转化为tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.labels[0]
    
    def __getitem__(self, idx):
        # shape: (n_features,seq_len) shape: ()
        return self.features[idx], self.labels[idx]
            
    def gen_train_test_ds(dataset):
        # 划分训练集测试集
        indices = np.arange(dataset.labels.shape[0])
        train_indices,test_indices = train_test_split(
            indices,
            test_size=dataset.CONFSP.test_ratio,
            stratify=dataset.labels
        )
        train_ds = Subset(dataset,train_indices)
        test_ds = Subset(dataset,test_indices)
        return train_ds, test_ds


if __name__ == "__main__":
    from config_sp import Config_SP
    CONFSP = Config_SP(T=180)
    ds = MultivariateTSDataset(CONFSP=CONFSP)
    train_ds, test_ds = MultivariateTSDataset.gen_train_test_ds(ds)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=False)
    for i, (x,y) in enumerate(train_loader):
        print(y)
        break
    