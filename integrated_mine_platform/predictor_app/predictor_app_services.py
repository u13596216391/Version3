import threading
import uuid
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from django.conf import settings
from .models import TrainingRun, ModelResult

# 1. 导入您的模型库
from .models_lib.lstm import LSTMPredictor, GRUPredictor, TransformerPredictor
from .models_lib.hybrid_models import (CNNLSTMPredictor, CNNGRUPredictor, LightGBMWrapper, DLinear, TimesNet, NHiTSWrapper)
from .models_lib.multi_scale_cnn import MultiScaleCNNEnsemble
from .models_lib.mamba import MambaPredictor

# 2. 迁移您 main.py 中的工具函数
def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'MSE': np.mean((y_true - y_pred) ** 2),
        'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'R2': r2_score(y_true, y_pred)
    }
    non_zero_mask = y_true != 0
    metrics['MAPE'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 if np.any(non_zero_mask) else np.inf
    return metrics

# 3. 核心训练服务类
class PredictionTrainingService:
    def __init__(self, task_id, config):
        self.task_id = task_id
        self.config = config
        self.run_instance = TrainingRun.objects.get(task_id=self.task_id)

    def _update_status(self, status, message):
        self.run_instance.status = status
        self.run_instance.status_message = message
        self.run_instance.save()

    def _get_model_instance(self, model_name):
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
            self._update_status(TrainingRun.Status.RUNNING, "任务开始，正在准备数据...")
            
            # 数据准备
            cfg = self.config['hyperparameters']
            df = pd.read_csv(self.config['data']['file_path'])
            raw_data = df.iloc[:, -1].values # 假设目标值在最后一列
            
            X, y = [], []
            for i in range(len(raw_data) - cfg['window_size']):
                X.append(raw_data[i:i+cfg['window_size']])
                y.append(raw_data[i+cfg['window_size']])
            
            X, y = np.array(X).reshape(-1, cfg['window_size'], 1), np.array(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['test_size'], shuffle=False)
            
            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=cfg['batch_size'], shuffle=False)

            # 循环训练模型
            for i, model_name in enumerate(self.config['models']):
                self._update_status(TrainingRun.Status.RUNNING, f"({i+1}/{len(self.config['models'])}) 正在训练 {model_name}...")
                model = self._get_model_instance(model_name)
                
                start_time = time.time()
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters())
                
                losses = []
                for epoch in range(cfg['epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        loss.backward()
                        optimizer.step()
                    losses.append(loss.item())
                
                training_time = time.time() - start_time
                
                model.eval()
                predictions, actuals = [], []
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        predictions.extend(outputs.squeeze().cpu().numpy())
                        actuals.extend(batch_y.cpu().numpy())
                
                metrics = calculate_metrics(actuals, predictions)

                # 保存结果
                results_dir = os.path.join(settings.MEDIA_ROOT, 'prediction_results', str(self.task_id))
                os.makedirs(results_dir, exist_ok=True)
                
                preds_path = os.path.join(results_dir, f"{model_name}_preds.npz")
                np.savez(preds_path, predictions=np.array(predictions), actuals=np.array(actuals))
                
                weights_path = os.path.join(results_dir, f"{model_name}.pth")
                torch.save(model.state_dict(), weights_path)
                
                ModelResult.objects.create(
                    training_run=self.run_instance, model_name=model_name, metrics=metrics,
                    training_time_seconds=training_time, loss_curve=losses,
                    predictions_file_path=preds_path, model_weights_path=weights_path
                )

            self._update_status(TrainingRun.Status.SUCCESS, "所有模型训练完成！")
        except Exception as e:
            import traceback
            self.run_instance.error_message = traceback.format_exc()
            self._update_status(TrainingRun.Status.FAILED, f"任务失败: {e}")
        finally:
            self.run_instance.completed_at = datetime.now()
            self.run_instance.save()

def start_training_session_async(config, temp_file_path):
    task_id = uuid.uuid4()
    run = TrainingRun.objects.create(task_id=task_id, status=TrainingRun.Status.PENDING, training_config=config)
    config['data'] = {'file_path': temp_file_path}
    
    thread = threading.Thread(target=PredictionTrainingService(task_id=run.task_id, config=config).run)
    thread.daemon = True
    thread.start()
    return task_id
