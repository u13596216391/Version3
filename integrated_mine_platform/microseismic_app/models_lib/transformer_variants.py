import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # ... (此辅助类保持不变)
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class BasicTransformerPredictor(nn.Module):
    """
    简化的基础 Transformer 预测模型
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, time_steps, output_dim, **kwargs):
        super().__init__()
        
        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=time_steps)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Linear(d_model, output_dim)
    
    def forward(self, features, **kwargs):
        """
        简化的前向传播
        参数:
            features: [batch_size, time_steps, input_dim]
        """
        embedded = self.feature_embedding(features)
        embedded = self.pos_encoder(embedded)
        encoded = self.transformer_encoder(embedded)
        last_hidden = encoded[:, -1, :]
        prediction = self.predictor(last_hidden)
        return prediction.squeeze(-1)
