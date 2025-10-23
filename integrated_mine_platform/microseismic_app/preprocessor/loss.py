import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss

class MicroseismicLoss(_Loss):
    """
    微震能量预测模型的综合损失函数
    结合MSE、MAE和注意力正则化
    """
    
    def __init__(self, mse_weight=1.0, mae_weight=0.5, attention_weight=0.1, 
                smooth_l1_weight=0.3, reduction='mean'):
        """
        初始化损失函数
        
        参数:
            mse_weight (float): MSE损失权重
            mae_weight (float): MAE损失权重
            attention_weight (float): 注意力正则化权重
            smooth_l1_weight (float): Smooth L1损失权重
            reduction (str): 'mean', 'sum', 'none'
        """
        super(MicroseismicLoss, self).__init__(reduction=reduction)
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.attention_weight = attention_weight
        self.smooth_l1_weight = smooth_l1_weight
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.mae_loss = nn.L1Loss(reduction=reduction)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction=reduction)
    
    def forward(self, pred, target, attention_weights=None):
        """
        计算损失
        
        参数:
            pred (Tensor): 预测值 [batch_size, 1]
            target (Tensor): 目标值 [batch_size, 1]
            attention_weights (dict): 注意力权重字典
            
        返回:
            loss (Tensor): 综合损失值
            loss_dict (dict): 各部分损失组成
        """
        # 计算MSE损失
        mse = self.mse_loss(pred, target)
        
        # 计算MAE损失
        mae = self.mae_loss(pred, target)
        
        # 计算Smooth L1损失
        smooth_l1 = self.smooth_l1_loss(pred, target)
        
        # 注意力正则化损失
        attention_loss = torch.tensor(0.0, device=pred.device)
        
        if attention_weights is not None and self.attention_weight > 0:
            # 对注意力权重进行正则化，鼓励关注不同的时间点
            for key, attn in attention_weights.items():
                if 'micro_attention' in key or 'time_attention' in key:
                    # 处理时序注意力，鼓励分散注意力到不同时间点
                    if attn.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
                        # 对于每个头的注意力，计算熵
                        for head_idx in range(attn.size(1)):
                            head_attn = attn[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]
                            # 取最后一个时间步的注意力分布
                            last_step_attn = head_attn[:, -1, :]  # [batch_size, seq_len]
                            # 计算熵，高熵表示注意力更加分散
                            entropy = -torch.sum(last_step_attn * torch.log(last_step_attn + 1e-10), dim=1).mean()
                            # 低熵应该被惩罚，添加负熵作为损失
                            attention_loss = attention_loss - entropy / attn.size(1)
                    elif attn.dim() == 2:  # [batch_size, seq_len]
                        # 简化版本，直接计算熵
                        entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=1).mean()
                        attention_loss = attention_loss - entropy
        
        # 计算总损失
        total_loss = (
            self.mse_weight * mse + 
            self.mae_weight * mae + 
            self.smooth_l1_weight * smooth_l1 + 
            self.attention_weight * attention_loss
        )
        
        # 创建损失字典，用于记录
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse.item(),
            'mae_loss': mae.item(),
            'smooth_l1_loss': smooth_l1.item(),
            'attention_loss': attention_loss.item()
        }
        
        return total_loss, loss_dict

class FocalMSELoss(_Loss):
    """
    焦点MSE损失函数，针对大误差样本给予更高权重
    """
    
    def __init__(self, alpha=2.0, reduction='mean'):
        """
        初始化焦点MSE损失
        
        参数:
            alpha (float): 聚焦参数，较大值会更加关注难分类样本
            reduction (str): 'mean', 'sum', 'none'
        """
        super(FocalMSELoss, self).__init__(reduction=reduction)
        self.alpha = alpha
        
    def forward(self, pred, target):
        """
        计算焦点MSE损失
        
        参数:
            pred (Tensor): 预测值 [batch_size, 1]
            target (Tensor): 目标值 [batch_size, 1]
            
        返回:
            loss (Tensor): 损失值
        """
        # 计算MSE
        mse = F.mse_loss(pred, target, reduction='none')
        
        # 计算权重，误差越大，权重越高
        weights = torch.pow(mse, self.alpha / 2.0)
        
        # 应用权重
        focal_mse = weights * mse
        
        # 应用缩减
        if self.reduction == 'mean':
            return focal_mse.mean()
        elif self.reduction == 'sum':
            return focal_mse.sum()
        else:  # 'none'
            return focal_mse
