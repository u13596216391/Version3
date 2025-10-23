import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size,
            padding=(kernel_size-1)*dilation//2,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MultiScaleConvPath(nn.Module):
    def __init__(self, in_channels, kernel_size, dilations=[1, 2, 4]):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            DilatedConvBlock(in_channels, 32, kernel_size, d)
            for d in dilations
        ])
        
    def forward(self, x):
        outputs = [block(x) for block in self.conv_blocks]
        return torch.cat(outputs, dim=1)

class AutoRegressive(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(seq_len))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.squeeze(-1)  # Remove feature dimension
        return torch.sum(x * self.weights, dim=1) + self.bias

class MultiScaleCNNEnsemble(nn.Module, BaseModel):
    def __init__(self, input_size=1, window_size=10):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "MultiScaleCNN-AR"
        self.window_size = window_size
        # 长期建模分支
        self.long_term = MultiScaleConvPath(
            input_size, 
            kernel_size=5,
            dilations=[1, 2, 4, 8]
        )
        
        # 短期建模分支
        self.short_term = MultiScaleConvPath(
            input_size,
            kernel_size=3,
            dilations=[1, 2, 4]
        )
        
        # 自回归模块
        self.ar = AutoRegressive(window_size)
        
        # 融合层
        conv_out_size = (32 * 4) + (32 * 3)  # 长期4个分支 + 短期3个分支
        self.fusion = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Ensemble权重
        self.ensemble_weight = nn.Parameter(torch.tensor([0.7, 0.3]))
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x_conv = x.transpose(1, 2)  # 转换为conv1d需要的形状
        
        # 长短期特征提取
        long_features = self.long_term(x_conv)
        short_features = self.short_term(x_conv)
        
        # 特征融合
        combined = torch.cat([long_features, short_features], dim=1)
        combined = combined.mean(dim=2)  # 时间维度平均池化
        cnn_pred = self.fusion(combined)
        
        # 自回归预测
        ar_pred = self.ar(x).unsqueeze(-1)
        
        # Ensemble
        weights = F.softmax(self.ensemble_weight, dim=0)
        final_pred = weights[0] * cnn_pred + weights[1] * ar_pred
        
        return final_pred.squeeze()

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, pred, target):
        loss = torch.zeros_like(pred)
        condition = torch.abs(pred - target) <= self.delta
        loss[condition] = 0.5 * (pred[condition] - target[condition])**2
        loss[~condition] = self.delta * torch.abs(pred[~condition] - 
                          target[~condition]) - 0.5 * self.delta**2
        return loss.mean()