import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
import os

class MicroseismicSequenceDataset(Dataset):
    """为简化的微震数据创建样本"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # 返回一个字典，即使只有一个特征，也保持良好实践
        return {
            'features': self.X[idx],
            'target': self.y[idx]
        }

def create_dataloaders(config, processed_data_path, run_dir):
    """加载简化的预处理数据并创建DataLoader"""
    print("\n--- 创建简化版数据加载器 ---")
    
    data_path = os.path.join(processed_data_path, 'data.joblib')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"预处理数据文件未找到: {data_path}")
        
    data = joblib.load(data_path)
    (X_train, y_train) = data['train']
    (X_val, y_val) = data['val']
    (X_test, y_test) = data['test']

    print(f"从 {data_path} 加载数据。")
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

    batch_size = config.get('batch_size', 32)

    train_dataset = MicroseismicSequenceDataset(X_train, y_train)
    val_dataset = MicroseismicSequenceDataset(X_val, y_val)
    test_dataset = MicroseismicSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("数据加载器创建完成。")
    return (train_loader, val_loader, test_loader), None, None # 返回None以匹配旧接口