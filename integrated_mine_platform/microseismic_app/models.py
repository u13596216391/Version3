import uuid
from django.db import models

class MSTrainingRun(models.Model):
    # ... (此模型保持不变)
    class Status(models.TextChoices):
        PENDING = 'PENDING', '待处理'
        RUNNING = 'RUNNING', '正在运行'
        SUCCESS = 'SUCCESS', '成功'
        FAILED = 'FAILED', '失败'
    task_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    training_config = models.JSONField(default=dict)
    status_message = models.CharField(max_length=255, default="任务已创建", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)

class MSModelResult(models.Model):
    training_run = models.ForeignKey(MSTrainingRun, related_name='model_results', on_delete=models.CASCADE)
    model_name = models.CharField(max_length=100)
    metrics = models.JSONField(default=dict)
    
    # --- 新增字段，用于存储图表数据文件的路径 ---
    plot_file_path = models.CharField(max_length=255, blank=True, null=True)
