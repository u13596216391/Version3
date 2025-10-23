import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations  # 更新导入
from .BaseModel import BaseModel

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)
        # CNN需要: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # 转回RNN需要的形状: (batch, seq_len, features)
        return x.transpose(1, 2)

class CNNLSTMPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-LSTM"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze()

class CNNGRUPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-GRU"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.gru = nn.GRU(64, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze()

class CNNBiLSTMPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-BiLSTM"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.bilstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2因为是双向
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.bilstm(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze()

class CNNBiGRUPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-BiGRU"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.bigru = nn.GRU(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2因为是双向
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.bigru(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze()

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 应用注意力权重并聚合时间步
        context = torch.bmm(attention_weights.transpose(-2, -1), x)
        return context.squeeze(1)

class CNNLSTMAttentionPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-LSTM-Attention"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return x.squeeze()

class CNNGRUAttentionPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-GRU-Attention"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.gru = nn.GRU(64, hidden_size, num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.gru(x)
        x = self.attention(x)
        x = self.fc(x)
        return x.squeeze()

class CNNBiLSTMAttentionPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-BiLSTM-Attention"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.bilstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.bilstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return x.squeeze()
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        
        # 计算padding以保持输出尺寸与输入相同
        self.padding = ((kernel_size-1) * dilation) // 2
        
        self.conv1 = parametrizations.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                     stride=stride, padding=self.padding, dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = parametrizations.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                     stride=stride, padding=self.padding, dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        
        # 确保输出维度匹配
        if out.size(2) != res.size(2):
            # 调整padding
            diff = out.size(2) - res.size(2)
            if diff > 0:
                res = F.pad(res, (0, diff))
            else:
                out = F.pad(out, (0, -diff))
                
        return self.relu(out + res)


class CNNBiGRUAttentionPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "CNN-BiGRU-Attention"
        self.feature_extractor = CNNFeatureExtractor(input_size)
        self.bigru = nn.GRU(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.bigru(x)
        x = self.attention(x)
        x = self.fc(x)
        return x.squeeze()

class TCNAttentionPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, num_channels=[64,64], kernel_size=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "TCN-Attention"
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                in_channels, 
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=((kernel_size-1) * dilation_size) // 2
            )]
        
        self.tcn = nn.Sequential(*layers)
        self.attention = AttentionLayer(num_channels[-1])
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        # [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # TCN处理
        x = self.tcn(x)  # [batch, channels, seq_len]
        
        # [batch, channels, seq_len] -> [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # 应用注意力机制得到上下文向量
        x = self.attention(x)  # [batch, channels]
        
        # 全连接层预测
        x = self.fc(x)  # [batch, 1]
        return x.squeeze()
    # hybrid_models.py 新增内容
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .BaseModel import BaseModel
from torch.nn.utils import weight_norm
import lightgbm as lgb
from pytorch_forecasting.models import NHiTS, TemporalFusionTransformer

class DLinear(nn.Module, BaseModel):
    """DLinear 模型 (ICLR 2023)"""
    def __init__(self, seq_len=6, pred_len=1, hidden_size=64):  # 新增参数
        super().__init__()
        self.name = "DLinear"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        
        # 移动平均窗口
        kernel_size = min(3, seq_len)
        padding = (kernel_size - 1) // 2
        
        # 趋势分解层
        self.decomp = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding),
            nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        )
        
        # 预测层
        self.trend_predictor = nn.Sequential(
            nn.Linear(seq_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.seasonal_predictor = nn.Sequential(
            nn.Linear(seq_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        x_in = x.transpose(1, 2)
        trend = self.decomp(x_in).transpose(1, 2)
        seasonal = x - trend
        
        trend = trend.squeeze(-1)
        seasonal = seasonal.squeeze(-1)
        
        trend_pred = self.trend_predictor(trend)
        seasonal_pred = self.seasonal_predictor(seasonal)
        return (trend_pred + seasonal_pred).squeeze(-1)

class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k=5):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        
        # 2D卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            # (batch, seq_len) -> (batch, seq_len, 1)
            x = x.unsqueeze(-1)
        
        # 现在 x 的形状应该是 (batch, seq_len, features)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        features = x.shape[2]
        
        # 重塑为图像格式 (batch, channels, height, width)
        x_2d = x.reshape(batch_size, 1, features, seq_len)
        
        # 应用2D卷积
        conv_out = self.conv(x_2d)  # (batch, 1, features, seq_len)
        
        # 重塑回序列格式 (batch, seq_len, features)
        out = conv_out.permute(0, 3, 2, 1).squeeze(-1)
        
        return out

class TimesNet(nn.Module, BaseModel):
    def __init__(self, seq_len=6, pred_len=1, top_k=5):
        super().__init__()
        self.name = "TimesNet"
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 输入层
        self.enc_embedding = nn.Linear(1, 16)
        
        # TimesBlock层
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k)
            for _ in range(2)
        ])
        
        # 输出层
        self.projection = nn.Linear(16, 1)
        
    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
            
        # 嵌入
        x = self.enc_embedding(x)  # (batch, seq_len, d_model)
        
        # 通过TimesBlock
        for block in self.blocks:
            x = block(x)
            
        # 预测
        x = self.projection(x)  # (batch, seq_len, 1)
        x = x[:, -self.pred_len:, :]  # 只返回预测长度的输出
        
        # 调整输出维度以匹配目标
        x = x.squeeze()  # 移除所有维度为1的维度
        if len(x.shape) == 0:  # 如果输出变成标量
            x = x.unsqueeze(0)  # 添加batch维度
            
        return x

class NHiTSWrapper(nn.Module, BaseModel):
    def __init__(self, seq_len=6, pred_len=1):
        super().__init__()
        BaseModel.__init__(self)
        self.name = "N-HiTS"
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 简化的基础网络结构
        self.backbone = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len)
        )
    
    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 3:
            x = x.squeeze(-1)
        return self.backbone(x)

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
import pytorch_lightning as pl

class TFTPredictor(nn.Module, BaseModel):
    def __init__(self, seq_len=6, pred_len=1):
        super().__init__()
        BaseModel.__init__(self)
        self.name = "TFT"
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 添加基础网络层
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, features)
        batch_size = x.shape[0]
        
        # 处理输入
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
            
        # 编码
        x = x.view(batch_size, -1)
        x = self.encoder(x)
        x = x.unsqueeze(1)  # 为注意力机制添加序列维度
        
        # 应用时间注意力
        attn_output, _ = self.temporal_attn(x, x, x)
        
        # 解码并预测
        out = self.decoder(attn_output.squeeze(1))
        return out.squeeze()
class LightGBMWrapper(nn.Module, BaseModel):
    """LightGBM 模型包装类"""
    def __init__(self, input_size=1, window_size=6, **kwargs):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "LightGBM"
        self.window_size = window_size
        self.model = None
        self.params = {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'objective': 'regression',
            'verbose': -1,
            **kwargs
        }
        
    def fit(self, X_train, y_train):
        """训练模型"""
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_flat, y_train)
        return self
        
    def forward(self, x):
        """模型预测"""
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit方法")
            
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        x_flat = x.reshape(x.shape[0], -1)
        predictions = self.model.predict(x_flat)
        return torch.FloatTensor(predictions)
    