from django.db import models
from django.utils import timezone


class MicroseismicEvent(models.Model):
    """微震事件数据模型"""
    DATA_TYPES = [
        ('frequency', '频次'),
        ('energy', '能量'),
    ]
    
    timestamp = models.DateTimeField(verbose_name='时间戳', db_index=True)
    x_coord = models.FloatField(verbose_name='X坐标')
    y_coord = models.FloatField(verbose_name='Y坐标')
    z_coord = models.FloatField(verbose_name='Z坐标', null=True, blank=True)
    energy = models.FloatField(verbose_name='能量', null=True, blank=True)
    magnitude = models.FloatField(verbose_name='震级', null=True, blank=True)
    data_type = models.CharField(max_length=20, choices=DATA_TYPES, default='frequency', verbose_name='数据类型')
    is_simulated = models.BooleanField(default=False, verbose_name='是否为模拟数据', db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        db_table = 'analysis_microseismic_event'
        verbose_name = '微震事件'
        verbose_name_plural = '微震事件'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp', 'data_type']),
        ]

    def __str__(self):
        return f"微震事件 {self.timestamp} ({self.x_coord}, {self.y_coord})"


class SupportResistance(models.Model):
    """支架阻力数据模型"""
    timestamp = models.DateTimeField(verbose_name='时间戳', db_index=True)
    station_id = models.CharField(max_length=50, verbose_name='测站ID', db_index=True)
    resistance_value = models.FloatField(verbose_name='阻力值(MPa)')
    pressure_level = models.CharField(max_length=20, verbose_name='压力等级', null=True, blank=True)
    is_abnormal = models.BooleanField(default=False, verbose_name='是否异常', db_index=True)
    is_simulated = models.BooleanField(default=False, verbose_name='是否为模拟数据', db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        db_table = 'analysis_support_resistance'
        verbose_name = '支架阻力'
        verbose_name_plural = '支架阻力'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp', 'station_id']),
        ]

    def __str__(self):
        return f"{self.station_id} - {self.timestamp}: {self.resistance_value}MPa"


class ProgressData(models.Model):
    """生产进尺数据模型"""
    date = models.DateField(verbose_name='日期', db_index=True)
    progress = models.FloatField(verbose_name='进尺(米)')
    work_face = models.CharField(max_length=50, verbose_name='工作面', default='主工作面')
    notes = models.TextField(verbose_name='备注', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        db_table = 'analysis_progress_data'
        verbose_name = '生产进尺'
        verbose_name_plural = '生产进尺'
        ordering = ['-date']
        unique_together = [['date', 'work_face']]

    def __str__(self):
        return f"{self.work_face} - {self.date}: {self.progress}m"


class AnalysisResult(models.Model):
    """分析结果缓存模型"""
    ANALYSIS_TYPES = [
        ('microseismic_scatter', '微震散点图'),
        ('microseismic_density', '微震核密度图'),
        ('support_dwt', '支架阻力DWT分析'),
        ('support_wavelet', '支架阻力小波分析'),
        ('correlation', '关联性分析'),
        ('hard_roof', '坚硬顶板分析'),
    ]
    
    analysis_type = models.CharField(max_length=50, choices=ANALYSIS_TYPES, verbose_name='分析类型', db_index=True)
    parameters = models.JSONField(verbose_name='分析参数')
    result_data = models.JSONField(verbose_name='结果数据')
    image_data = models.TextField(verbose_name='图表数据(Base64)', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间', db_index=True)
    expires_at = models.DateTimeField(verbose_name='过期时间', null=True, blank=True)

    class Meta:
        db_table = 'analysis_result'
        verbose_name = '分析结果'
        verbose_name_plural = '分析结果'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_analysis_type_display()} - {self.created_at}"

    def is_expired(self):
        """检查结果是否过期"""
        if not self.expires_at:
            return False
        return timezone.now() > self.expires_at
