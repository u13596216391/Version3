import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(d_model, d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
    
    def forward(self, u):
        B, L, D = u.shape
        # 初始化状态 (B, D, d_state)
        state = torch.zeros(B, D, self.d_state, device=u.device)
        
        # 预计算所有时间步的输入扩展
        u_expanded = u.unsqueeze(-1)  # (B, L, D, 1)
        
        # 批量计算状态更新
        states = []
        for t in range(L):
            # 向量化计算: state = state @ A + u[:, t] * B
            state = torch.einsum('bdi,dij->bdj', state, self.A) + u_expanded[:, t] * self.B
            state = torch.tanh(state)
            states.append(state)
        
        # 合并所有时间步的状态
        states = torch.stack(states, dim=1)  # (B, L, D, d_state)
        
        # 计算输出: y = u + sum(state * C)
        y = u + torch.einsum('blds,ds->bld', states, self.C)
        return y

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # 残差连接
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.proj(x)
        return x + residual

class MambaPredictor(nn.Module, BaseModel):
    def __init__(self, input_size=1, d_model=64, num_layers=4):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.name = "SimplifiedMamba"
        
        # 输入投影
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Mamba层
        self.layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(num_layers)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # 通过Mamba层
        for layer in self.layers:
            x = layer(x)
            
        # 输出预测
        x = self.norm(x)
        x = self.fc(x[:, -1])  # 使用序列最后一个时间步
        return x.squeeze()