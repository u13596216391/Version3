import torch
import torch.nn as nn
from .BaseModel import BaseModel

class LSTMPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)  # 首先初始化nn.Module
        BaseModel.__init__(self)  # 然后初始化BaseModel
        self.name = "LSTM"
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions.squeeze()

class GRUPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "GRU"
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        predictions = self.fc(gru_out[:, -1, :])
        return predictions.squeeze()

class TransformerPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, d_model=64, nhead=8, num_layers=2):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "Transformer"
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers  # 使用传入的num_layers参数
        )
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        predictions = self.fc(x[:, -1, :])
        return predictions.squeeze()
