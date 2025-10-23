from django.db import models
from django.utils import timezone


class MonitoringData(models.Model):
    """监控数据模型"""
    
    DATA_TYPES = [
        ('microseismic', '微震数据'),
        ('support_resistance', '支架阻力'),
        ('gas', '瓦斯浓度'),
        ('temperature', '温度'),
        ('humidity', '湿度'),
    ]
    
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    data_type = models.CharField(max_length=50, choices=DATA_TYPES, verbose_name='数据类型')
    location = models.CharField(max_length=100, verbose_name='监测位置')
    value = models.FloatField(verbose_name='数值')
    unit = models.CharField(max_length=20, verbose_name='单位')
    is_abnormal = models.BooleanField(default=False, verbose_name='是否异常')
    
    class Meta:
        verbose_name = '监控数据'
        verbose_name_plural = '监控数据'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['data_type', 'timestamp']),
            models.Index(fields=['location', 'timestamp']),
        ]
    
    def __str__(self):
        return f'{self.data_type} - {self.location} - {self.value}{self.unit}'
