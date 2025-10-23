import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, output_dim, **kwargs):
        """
        简化的 LSTM 预测模型
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, features, **kwargs):
        """
        简化的前向传播，只接收一个 'features' 张量
        参数:
            features: [batch_size, time_steps, input_dim]
        """
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # 使用最后一个时间步的输出来进行预测
        last_time_step_out = lstm_out[:, -1, :]
        
        prediction = self.output_layer(self.dropout(last_time_step_out))
        
        return prediction.squeeze(-1) # 确保输出维度正确
