# microseismic_app/services.py
import threading
import uuid
import os
import time
import yaml
import zipfile
import tempfile
import joblib
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import traceback
import matplotlib.pyplot as plt

from django.conf import settings
from .models import MSTrainingRun, MSModelResult

# ===================================================================
# 1. 导入您的预处理、数据加载器和损失函数
# ===================================================================
from .preprocessor.preprocess import process_data
from .preprocessor.dataloader import create_dataloaders
from .preprocessor.loss import MicroseismicLoss

# ===================================================================
# 2. 导入您的所有模型库
# ===================================================================
from .models_lib.transformer_variants import DualPathTransformerPredictor, InformerPredictor, BasicTransformerPredictor
from .models_lib.lstm_model import LSTMPredictor
from .models_lib.classic_models import XGBoostModel
from .models_lib.MambaPredictor import MambaPredictor

# ===================================================================
# 3. 迁移 train.py 中的核心工具函数
#    我们将它们作为Service类的私有方法，以便更好地封装
# ===================================================================

class MicroseismicTrainingService:
    def __init__(self, task_id, config, file_paths):
        self.task_id = task_id
        self.config = config
        self.file_paths = file_paths
        self.run_instance = MSTrainingRun.objects.get(task_id=self.task_id)
        self.run_dir = os.path.join(settings.MEDIA_ROOT, 'microseismic_results', str(self.task_id))
        os.makedirs(self.run_dir, exist_ok=True)
        self.device = torch.device(config.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')

    # --- 状态与日志更新 ---
    def _update_status(self, status, message):
        self.run_instance.status = status
        self.run_instance.status_message = message
        self.run_instance.save()
    
    # --- 模型初始化 ---
    def _get_model_instance(self):
        """根据config中的model_type动态初始化模型实例"""
        model_type = self.config['model_type']
        model_map = {
            'dual_path': DualPathTransformerPredictor,
            'informer': InformerPredictor,
            'lstm': LSTMPredictor,
            'base_transformer': BasicTransformerPredictor,
            'mamba': MambaPredictor,
            'xgboost': XGBoostModel,
        }
        model_class = model_map.get(model_type)
        if not model_class:
            raise ValueError(f"未知的模型类型: {model_type}")

        # 为不同模型准备参数
        if model_type == 'xgboost':
            return model_class(self.config.get('xgboost_params', {}))
        
        # PyTorch模型的通用参数
        model_params = {
            'time_steps': self.config['time_steps'],
            'd_model': self.config['d_model'],
            'nhead': self.config['nhead'],
            'num_layers': self.config['num_layers'],
            'dropout': self.config['dropout'],
            'num_timesteps_output': self.config.get('num_timesteps_output', 1)
        }
        if model_type == 'mamba':
             model_params.update(self.config.get('mamba_params', {}))

        return model_class(**model_params)

    # --- PyTorch训练与评估步骤 ---
    def _train_step(self, model, dataloader, loss_fn, optimizer):
        model.train()
        total_loss = 0
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            optimizer.zero_grad()
            target = batch.pop('target')
            pred, attention = model(**batch)
            loss, _ = loss_fn(pred.squeeze(), target.squeeze(), attention)
            
            if torch.isnan(loss):
                raise ValueError("训练中出现NaN损失，任务终止。")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _eval_step(self, model, dataloader, loss_fn):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                target = batch.pop('target')
                pred, attention = model(**batch)
                loss, _ = loss_fn(pred.squeeze(), target.squeeze(), attention)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def _evaluate_and_visualize(self, model, test_loader, scaler):
        """在测试集上评估并生成可视化结果，返回评估指标"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        model.load_state_dict(torch.load(os.path.join(self.run_dir, 'best_model.pth')))
        model.to(self.device)
        model.eval()
        
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                pred, _ = model(**{k: v for k, v in batch.items() if k != 'target'})
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch['target'].cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()

        # 反归一化
        dummy_preds = np.zeros((len(all_preds), 3))
        dummy_targets = np.zeros((len(all_targets), 3))
        dummy_preds[:, 2] = all_preds
        dummy_targets[:, 2] = all_targets
        
        log_preds = scaler.inverse_transform(dummy_preds)[:, 2]
        log_targets = scaler.inverse_transform(dummy_targets)[:, 2]
        
        actual_preds = np.expm1(log_preds)
        actual_targets = np.expm1(log_targets)

        # 计算指标
        metrics = {
            'mse': mean_squared_error(actual_targets, actual_preds),
            'rmse': np.sqrt(mean_squared_error(actual_targets, actual_preds)),
            'mae': mean_absolute_error(actual_targets, actual_preds),
            'r2': r2_score(actual_targets, actual_preds)
        }

        # 可视化：预测值 vs 真实值散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(actual_targets, actual_preds, alpha=0.5, s=10)
        plt.plot([min(actual_targets), max(actual_targets)], [min(actual_targets), max(actual_targets)], 'r--')
        plt.title('Prediction vs. Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        scatter_path = os.path.join(self.run_dir, 'prediction_vs_actual_scatter.png')
        plt.savefig(scatter_path)
        plt.close()
        
        return metrics, scatter_path

    # --- 主执行流程 ---
    def run(self):
        try:
            self._update_status(MSTrainingRun.Status.RUNNING, "任务开始，正在准备数据...")
            
            # 动态更新config中的文件路径
            for key, path in self.file_paths.items():
                if key in self.config['data_paths']:
                    self.config['data_paths'][key] = path
            
            # 1. 数据预处理
            process_data(self.config, self.run_dir)
            if not os.path.exists(os.path.join(self.run_dir, 'data.joblib')):
                 raise FileNotFoundError("数据预处理失败，未生成 'data.joblib' 文件。")

            # 2. 创建数据加载器
            data, _, _ = create_dataloaders(self.config, self.run_dir, self.run_dir)

            # 3. 初始化模型
            model = self._get_model_instance()
            model_type = self.config['model_type']

            # 4. 根据模型类型执行不同训练流程
            if model_type == 'xgboost':
                (X_train, y_train, X_val, y_val, X_test, y_test) = data
                self._update_status(MSTrainingRun.Status.RUNNING, "正在训练XGBoost模型...")
                model.train(X_train, y_train, X_val, y_val)
                model_path = os.path.join(self.run_dir, 'best_model.json')
                model.save(model_path)
                
                # 评估XGBoost
                from sklearn.metrics import mean_squared_error, r2_score
                preds = model.predict(X_test)
                metrics = {
                    'mse': float(mean_squared_error(y_test, preds)),
                    'r2': float(r2_score(y_test, preds))
                }
                # (此处可以添加XGBoost的可视化)
                
            else: # PyTorch模型
                (train_loader, val_loader, test_loader) = data
                model.to(self.device)
                
                loss_fn = MicroseismicLoss(**self.config.get('loss_weights', {}))
                optimizer = optim.Adam(model.parameters(), lr=float(self.config['learning_rate']), weight_decay=float(self.config['weight_decay']))
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config['patience'] // 2)

                best_val_loss = float('inf')
                epochs_no_improve = 0
                history = {'train_loss': [], 'val_loss': []}

                for epoch in range(self.config['epochs']):
                    self._update_status(MSTrainingRun.Status.RUNNING, f"训练中... Epoch {epoch+1}/{self.config['epochs']}")
                    
                    train_loss = self._train_step(model, train_loader, loss_fn, optimizer)
                    val_loss = self._eval_step(model, val_loader, loss_fn)
                    scheduler.step(val_loss)
                    
                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), os.path.join(self.run_dir, 'best_model.pth'))
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= self.config['patience']:
                        print(f"早停触发于 Epoch {epoch+1}")
                        break
                
                # 绘制损失曲线
                plt.figure(figsize=(10, 5))
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title('Training & Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                loss_curve_path = os.path.join(self.run_dir, 'loss_curve.png')
                plt.savefig(loss_curve_path)
                plt.close()

                # 最终评估
                scaler = joblib.load(os.path.join(self.run_dir, "scaler.joblib"))
                metrics, _ = self._evaluate_and_visualize(model, test_loader, scaler)

            # 5. 保存结果到数据库
            MSModelResult.objects.create(
                training_run=self.run_instance,
                model_name=model_type,
                metrics=metrics,
                # 可以添加更多字段来保存图表路径等
            )

            self._update_status(MSTrainingRun.Status.SUCCESS, "微震模型训练完成！")
        except Exception as e:
            error_msg = traceback.format_exc()
            self.run_instance.error_message = error_msg
            self._update_status(MSTrainingRun.Status.FAILED, f"任务失败: {e}")
        finally:
            self.run_instance.completed_at = datetime.now()
            self.run_instance.save()

def start_ms_training_async(config, file_paths):
    """异步启动训练的入口函数"""
    task_id = uuid.uuid4()
    run = MSTrainingRun.objects.create(task_id=task_id, status=MSTrainingRun.Status.PENDING, training_config=config)
    
    thread = threading.Thread(target=MicroseismicTrainingService(task_id=run.task_id, config=config, file_paths=file_paths).run)
    thread.daemon = True
    thread.start()
    return task_id
