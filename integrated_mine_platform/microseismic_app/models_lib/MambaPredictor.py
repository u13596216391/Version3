import torch
import torch.nn as nn
from .Mamba import Mamba, ModelArgs

class MambaPredictor(nn.Module):
    """
    简化的 Mamba 预测模型
    """
    def __init__(self, input_dim, d_model, n_layers, dropout, output_dim, d_state, expand, **kwargs):
        super().__init__()
        
        model_args = ModelArgs(
            d_model=d_model, n_layer=n_layers,
            vocab_size=1, # 对于连续数据不重要
            d_state=d_state, expand=expand, dropout=dropout
        )
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(model_args)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, features, **kwargs):
        """
        简化的前向传播
        参数:
            features: [batch_size, time_steps, input_dim]
        """
        # 1. 将输入投影到模型维度
        x = self.input_projection(features) # Shape: [batch, time_steps, d_model]
        
        # 2. 通过 Mamba 模型
        # Mamba.py 需要被修改以直接接受浮点数张量
        mamba_out = self.mamba(x) # Shape: [batch, time_steps, d_model]
        
        # 3. 使用最后一个时间步的输出进行预测
        last_hidden = mamba_out[:, -1, :] # Shape: [batch, d_model]
        
        # 4. 投影到最终输出维度
        prediction = self.output_projection(last_hidden) # Shape: [batch, output_dim]
        
        return prediction.squeeze(-1)
