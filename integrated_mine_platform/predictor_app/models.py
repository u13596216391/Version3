import uuid
from django.db import models

class TrainingRun(models.Model):
    """记录每一次训练会话的元数据。"""
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

    def __str__(self):
        return f"TrainingRun {self.task_id} - {self.status}"

class ModelResult(models.Model):
    """存储单次训练中，每一个模型的具体结果。"""
    training_run = models.ForeignKey(TrainingRun, related_name='model_results', on_delete=models.CASCADE)
    model_name = models.CharField(max_length=100)
    metrics = models.JSONField(default=dict)
    training_time_seconds = models.FloatField()
    loss_curve = models.JSONField(default=list)
    predictions_file_path = models.CharField(max_length=255)
    model_weights_path = models.CharField(max_length=255)
    plot_file_path = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.model_name} for run {self.training_run_id}"