import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torchsummary

class TimeSeriesCNN(nn.Module):
    """支持可变长度输入的多元时序CNN分类网络（保留4层卷积结构）"""
    def __init__(self, n_features, num_classes):
        super(TimeSeriesCNN, self).__init__()
        
        # 输入形状: (batch, n_features, seq_len)
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # seq_len -> seq_len//2
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)  # seq_len//2 -> seq_len//4
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)  # seq_len//4 -> seq_len//8
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化，适配任意长度
        )
        
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # 输入形状: (batch, n_features, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        return x
    
    def get_features(self, x):
        # 输入形状: (batch, n_features, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分类
        return x
        

class ResidualBlock(nn.Module):
    """残差块（适配时序数据）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size//2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=kernel_size//2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 捷径连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=stride
                ),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class TimeSeriesResNet(nn.Module):
    
    def __init__(self, n_features, num_classes):
        super(TimeSeriesResNet, self).__init__()
        
        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 180 -> 90
        )
        
        # 残差块组
        self.layer1 = self._make_layer(64, 64, 2, stride=1)  # 90 -> 90
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 90 -> 45
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 45 -> 22
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 22 -> 11
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """构建残差层"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 输入形状: (batch, n_features, 180)
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)  # (batch, 512, 1)
        x = x.view(x.size(0), -1)
        
        return self.classifier(x)

if __name__ == "__main__":
    # 初始化模型（无需指定长度）
    from config import CONF
    model = TimeSeriesResNet(n_features=CONF.n_features, num_classes=CONF.n_classes)
    torchsummary.summary(model,input_size=(38,180),device="cpu")
    #model = TimeSeriesCNN(n_features=CONF.n_features, num_classes=CONF.n_classes)

    # # 处理不同长度的输入
    # lengths = [30, 60, 120, 180, 240, 300]
    # for seq_len in lengths:
    #     x = torch.randn(32, CONF.n_features, seq_len)  # batch=32, features=10
    #     model.check_input_length(seq_len)  # 验证长度
    #     output = model(x)
    #     print(f"输入长度{seq_len} -> 输出形状: {output.shape}")