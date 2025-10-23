from django.db import models
from django.utils import timezone


class SystemStatistics(models.Model):
    """系统统计数据模型"""
    
    # 统计时间
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    
    # 任务统计
    total_tasks = models.IntegerField(default=0, verbose_name='总任务数')
    running_tasks = models.IntegerField(default=0, verbose_name='运行中任务')
    completed_tasks = models.IntegerField(default=0, verbose_name='已完成任务')
    failed_tasks = models.IntegerField(default=0, verbose_name='失败任务')
    
    # 预测统计
    predictor_tasks = models.IntegerField(default=0, verbose_name='支架阻力预测任务')
    microseismic_tasks = models.IntegerField(default=0, verbose_name='微震预测任务')
    
    # 数据统计
    total_data_records = models.IntegerField(default=0, verbose_name='数据记录总数')
    
    # 系统状态
    system_health = models.CharField(max_length=20, default='normal', verbose_name='系统健康状态')
    
    class Meta:
        verbose_name = '系统统计'
        verbose_name_plural = '系统统计'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f'Statistics at {self.timestamp}'


class AlertRecord(models.Model):
    """预警记录模型"""
    
    ALERT_LEVELS = [
        ('info', '信息'),
        ('warning', '警告'),
        ('danger', '危险'),
        ('critical', '严重'),
    ]
    
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    level = models.CharField(max_length=20, choices=ALERT_LEVELS, verbose_name='预警级别')
    title = models.CharField(max_length=200, verbose_name='预警标题')
    message = models.TextField(verbose_name='预警内容')
    source = models.CharField(max_length=100, verbose_name='预警来源')
    is_resolved = models.BooleanField(default=False, verbose_name='是否已处理')
    
    class Meta:
        verbose_name = '预警记录'
        verbose_name_plural = '预警记录'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f'{self.level} - {self.title}'
