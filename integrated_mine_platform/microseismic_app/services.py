# microseismic_app/services.py (Feature-Complete & Patched Version)

import threading
import uuid
import os
import time
import json
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import joblib
from django.conf import settings
from django.utils import timezone

from .models import MSTrainingRun, MSModelResult
from .preprocessor.preprocess import process_data
from .preprocessor.dataloader import create_dataloaders

# 导入模型库
from .models_lib.transformer_variants import BasicTransformerPredictor
from .models_lib.lstm_model import LSTMPredictor
from .models_lib.MambaPredictor import MambaPredictor

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    def clean_metric(value):
        if np.isinf(value) or np.isnan(value): return None
        return float(value)
    metrics = {
        'MAE': clean_metric(np.mean(np.abs(y_true - y_pred))),
        'MSE': clean_metric(np.mean((y_true - y_pred) ** 2)),
        'RMSE': clean_metric(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        'R2': clean_metric(r2_score(y_true, y_pred))
    }
    return metrics

class MicroseismicTrainingService:
    def __init__(self, task_id, config, file_paths):
        self.task_id = task_id
        self.config = config
        self.file_paths = file_paths
        self.run_instance = MSTrainingRun.objects.get(task_id=self.task_id)
        self.run_dir = os.path.join(settings.MEDIA_ROOT, 'microseismic_results', str(self.task_id))
        os.makedirs(self.run_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _update_status(self, status, message):
        self.run_instance.status = status
        self.run_instance.status_message = message
        self.run_instance.save()

    def _get_model_instance(self):
        """根据config动态初始化模型实例，并为不同模型适配正确的参数。"""
        model_type = self.config['model_type']
        model_map = {
            'lstm': LSTMPredictor,
            'base_transformer': BasicTransformerPredictor,
            'mamba': MambaPredictor,
        }
        model_class = model_map.get(model_type)
        if not model_class:
            raise ValueError(f"未知的模型类型: {model_type}")

        # --- 核心修复：为每个模型构建独立的、正确的参数字典 ---
        model_params = {}
        if model_type == 'lstm':
            # 根据您更新后的模型文件，LSTMPredictor 需要这些参数
            model_params = {
                'input_dim': 1, # 我们的简化版只有一个特征：能量
                'hidden_dim': self.config.get('d_model', 64), # 复用 d_model 作为 hidden_dim
                'n_layers': self.config.get('num_layers', 2),
                'dropout': self.config.get('dropout', 0.1),
                'output_dim': self.config.get('num_timesteps_output', 1)
            }
        elif model_type == 'base_transformer':
            # Transformer 通常需要知道序列长度(time_steps)和特征维度(d_model)
            model_params = {
                'input_dim': 1,
                'd_model': self.config.get('d_model', 64),
                'nhead': self.config.get('nhead', 4),
                'num_layers': self.config.get('num_layers', 2),
                'dropout': self.config.get('dropout', 0.1),
                'time_steps': self.config.get('time_steps', 24),
                'output_dim': self.config.get('num_timesteps_output', 1)
            }
        elif model_type == 'mamba':
            # Mamba 有自己的特定参数
            model_params = {
                'input_dim': 1,
                'd_model': self.config.get('d_model', 64),
                'n_layers': self.config.get('num_layers', 2),
                'dropout': self.config.get('dropout', 0.1),
                'output_dim': self.config.get('num_timesteps_output', 1),
                'd_state': self.config.get('d_state', 16),
                'expand': self.config.get('expand', 2)
            }
        
        return model_class(**model_params)

    def run(self):
        try:
            self._update_status(MSTrainingRun.Status.RUNNING, f"任务开始，使用设备: {self.device}")
            print(f"[{self.task_id}] 使用设备: {self.device}")

            # 1. 数据预处理
            self.config['data_paths'] = {'ms_folder': self.file_paths['ms_folder']}
            process_data(self.config, self.run_dir)

            # 2. 创建数据加载器
            (train_loader, val_loader, test_loader), _, _ = create_dataloaders(self.config, self.run_dir, self.run_dir)

            # 3. 初始化模型
            model = self._get_model_instance()
            model.to(self.device)
            
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config['learning_rate']))
            
            # 4. 训练循环
            for epoch in range(self.config['epochs']):
                self._update_status(MSTrainingRun.Status.RUNNING, f"训练中... Epoch {epoch+1}/{self.config['epochs']}")
                model.train()
                for batch in train_loader:
                    # 假设简化后的dataloader返回的batch是一个字典
                    features, target = batch['features'].to(self.device), batch['target'].to(self.device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs.squeeze(), target.squeeze())
                    loss.backward()
                    optimizer.step()

            # 5. 评估模型并反归一化
            model.eval()
            predictions_scaled, actuals_scaled = [], []
            with torch.no_grad():
                for batch in test_loader:
                    features, target = batch['features'].to(self.device), batch['target'].to(self.device)
                    outputs = model(features)
                    predictions_scaled.extend(outputs.squeeze().cpu().numpy().flatten())
                    actuals_scaled.extend(target.squeeze().cpu().numpy().flatten())
            
            scaler = joblib.load(os.path.join(self.run_dir, "scaler.joblib"))
            predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
            actuals = scaler.inverse_transform(np.array(actuals_scaled).reshape(-1, 1)).flatten()

            # 6. 计算指标
            metrics = calculate_metrics(actuals, predictions)

            # 7. 保存绘图用的JSON数据
            plot_data = {
                'actuals': [float(x) for x in actuals],
                'predictions': [float(x) for x in predictions]
            }
            plot_filename = f"{self.config['model_type']}_plot_data.json"
            plot_full_path = os.path.join(self.run_dir, plot_filename)
            with open(plot_full_path, 'w') as f:
                json.dump(plot_data, f)
            plot_relative_path = os.path.join('microseismic_results', str(self.task_id), plot_filename)

            # 8. 保存完整结果到数据库
            # 注意：这里的字段需要和 models.py 中的 MSModelResult 定义完全匹配
            MSModelResult.objects.create(
                training_run=self.run_instance,
                model_name=self.config['model_type'],
                metrics=metrics,
                plot_file_path=plot_relative_path
            )
            print(f"[{self.task_id}] 已成功保存 {self.config['model_type']} 的结果和绘图数据。")

            self._update_status(MSTrainingRun.Status.SUCCESS, "微震模型训练完成！")

        except Exception as e:
            error_msg = traceback.format_exc()
            self.run_instance.error_message = error_msg
            self._update_status(MSTrainingRun.Status.FAILED, f"任务失败: {e}")
            print(f"[{self.task_id}] 任务失败: {error_msg}")
        finally:
            self.run_instance.completed_at = timezone.now()
            self.run_instance.save()

def start_ms_training_async(config, file_paths):
    task_id = uuid.uuid4()
    run = MSTrainingRun.objects.create(task_id=task_id, status=MSTrainingRun.Status.PENDING, training_config=config)
    thread = threading.Thread(target=MicroseismicTrainingService(task_id=run.task_id, config=config, file_paths=file_paths).run)
    thread.daemon = True
    thread.start()
    return task_id
