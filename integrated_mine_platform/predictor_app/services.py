import threading
import uuid
import os
import time
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from django.utils import timezone
from django.conf import settings

# 配置日志
logger = logging.getLogger(__name__)

from .models import TrainingRun, ModelResult
from .models_lib.lstm import LSTMPredictor, GRUPredictor, TransformerPredictor
from .models_lib.hybrid_models import (CNNLSTMPredictor, CNNGRUPredictor, LightGBMWrapper, DLinear, TimesNet, NHiTSWrapper)
from .models_lib.multi_scale_cnn import MultiScaleCNNEnsemble
from .models_lib.mamba import MambaPredictor

def calculate_metrics(y_true, y_pred):
    # ... (此函数保持不变)
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
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        metrics['MAPE'] = clean_metric(mape)
    else:
        metrics['MAPE'] = None
    return metrics

class PredictionTrainingService:
    def __init__(self, task_id, config, temp_file_path):
        self.task_id = task_id
        self.config = config
        self.temp_file_path = temp_file_path
        self.run_instance = TrainingRun.objects.get(task_id=self.task_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _update_status(self, status, message):
        self.run_instance.status = status
        self.run_instance.status_message = message
        self.run_instance.save()

    def _get_model_instance(self, model_name):
        # ... (此函数保持不变)
        model_map = {
            "LSTM": LSTMPredictor, "GRU": GRUPredictor, "Transformer": TransformerPredictor,
            "CNN-LSTM": CNNLSTMPredictor, "CNN-GRU": CNNGRUPredictor,
            "MultiScaleCNN-AR": MultiScaleCNNEnsemble, "SimplifiedMamba": MambaPredictor,
            "LightGBM": LightGBMWrapper, "DLinear": DLinear, "TimesNet": TimesNet, "N-HiTS": NHiTSWrapper,
        }
        model_class = model_map.get(model_name)
        if not model_class: raise ValueError(f"未知模型: {model_name}")
        return model_class()

    def run(self):
        try:
            self._update_status(TrainingRun.Status.RUNNING, f"任务开始，正在使用设备: {self.device}")
            logger.info(f"[{self.task_id}] 使用设备: {self.device}")
            
            df = pd.read_csv(self.temp_file_path)
            logger.info(f"[{self.task_id}] 读取CSV文件，形状: {df.shape}, 列: {df.columns.tolist()}")
            
            # 根据数据类型选择目标列
            if 'magnitude' in df.columns:
                target_col = 'magnitude'
            elif 'energy' in df.columns:
                target_col = 'energy'
            elif 'resistance_value' in df.columns:
                target_col = 'resistance_value'
            else:
                # 如果没有明确的列，使用最后一列
                target_col = df.columns[-1]
            
            logger.info(f"[{self.task_id}] 使用目标列: {target_col}")
            raw_data = df[target_col].values
            logger.info(f"[{self.task_id}] 原始数据量: {len(raw_data)}, 数据类型: {raw_data.dtype}")
            logger.info(f"[{self.task_id}] 前5个值: {raw_data[:5]}")
            logger.info(f"[{self.task_id}] NaN数量: {pd.isna(raw_data).sum()}, Inf数量: {np.isinf(raw_data).sum()}")
            
            # 数据清洗：移除NaN和Inf值
            raw_data = pd.Series(raw_data).replace([np.inf, -np.inf], np.nan).dropna().values
            logger.info(f"[{self.task_id}] 清洗后数据量: {len(raw_data)}")
            
            if len(raw_data) == 0:
                raise ValueError("清洗后没有有效数据！所有数据都是NaN或Inf")
            
            logger.info(f"[{self.task_id}] 数据范围: [{raw_data.min():.2f}, {raw_data.max():.2f}]")
            
            if len(raw_data) < 100:
                raise ValueError(f"数据量不足（仅{len(raw_data)}条），至少需要100条有效数据")
            
            cfg = self.config['hyperparameters']
            X, y = [], []
            for i in range(len(raw_data) - cfg['window_size']):
                X.append(raw_data[i:i+cfg['window_size']])
                y.append(raw_data[i+cfg['window_size']])
            
            X, y = np.array(X).reshape(-1, cfg['window_size'], 1), np.array(y)
            
            # 检查是否有NaN
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("数据处理后仍包含NaN值")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['test_size'], shuffle=False)
            
            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=cfg['batch_size'], shuffle=False)

            for i, model_name in enumerate(self.config['models']):
                self._update_status(TrainingRun.Status.RUNNING, f"({i+1}/{len(self.config['models'])}) 正在训练 {model_name}...")
                
                model = self._get_model_instance(model_name)
                model.to(self.device)
                
                start_time = time.time()
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters())
                
                for epoch in range(cfg['epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        loss.backward()
                        optimizer.step()
                
                training_time = time.time() - start_time
                
                model.eval()
                predictions, actuals = [], []
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        predictions.extend(outputs.squeeze().cpu().numpy())
                        actuals.extend(batch_y.cpu().numpy())
                
                metrics = calculate_metrics(actuals, predictions)

                # --- 核心修改：生成并保存绘图用的JSON数据 ---
                results_dir = os.path.join(settings.MEDIA_ROOT, 'prediction_results', str(self.task_id))
                os.makedirs(results_dir, exist_ok=True)
                
                # 将numpy数组转换为python列表
                plot_data = {
                    'actuals': [float(x) for x in actuals],
                    'predictions': [float(x) for x in predictions]
                }
                
                # 保存数据到JSON文件
                plot_filename = f"{model_name}_plot_data.json"
                plot_full_path = os.path.join(results_dir, plot_filename)
                with open(plot_full_path, 'w') as f:
                    json.dump(plot_data, f)

                # 获取相对路径
                plot_relative_path = os.path.join('prediction_results', str(self.task_id), plot_filename)

                # --- 更新：保存结果到数据库，包含JSON数据路径 ---
                ModelResult.objects.create(
                    training_run=self.run_instance, 
                    model_name=model_name, 
                    metrics=metrics,
                    training_time_seconds=training_time,
                    plot_file_path=plot_relative_path # 保存JSON文件路径
                )
                logger.info(f"[{self.task_id}] 已成功保存 {model_name} 的结果和绘图数据。")

            self._update_status(TrainingRun.Status.SUCCESS, "所有模型训练完成！")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.run_instance.error_message = error_msg
            self._update_status(TrainingRun.Status.FAILED, f"任务失败: {e}")
            logger.error(f"[{self.task_id}] 任务失败: {error_msg}")
        finally:
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
            self.run_instance.completed_at = timezone.now()
            self.run_instance.save()

def start_training_session_async(config, temp_file_path):
    # ... (此函数保持不变)
    task_id = uuid.uuid4()
    run = TrainingRun.objects.create(task_id=task_id, status=TrainingRun.Status.PENDING, training_config=config)
    thread = threading.Thread(target=PredictionTrainingService(task_id=run.task_id, config=config, temp_file_path=temp_file_path).run)
    thread.daemon = True
    thread.start()
    return task_id
