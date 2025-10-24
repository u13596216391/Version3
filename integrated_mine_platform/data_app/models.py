from django.db import models
from django.utils import timezone


class UploadedFile(models.Model):
    """上传文件记录模型"""
    FILE_TYPES = [
        ('csv', 'CSV文件'),
        ('zip', 'ZIP压缩包'),
    ]
    
    DATA_TYPES = [
        ('microseismic', '微震数据'),
        ('support_resistance', '支架阻力'),
    ]
    
    filename = models.CharField(max_length=255, verbose_name='文件名')
    file_type = models.CharField(max_length=10, choices=FILE_TYPES, verbose_name='文件类型')
    data_type = models.CharField(max_length=50, choices=DATA_TYPES, verbose_name='数据类型')
    file_path = models.CharField(max_length=500, verbose_name='文件路径', null=True, blank=True)
    file_size = models.IntegerField(verbose_name='文件大小(字节)', default=0)
    upload_time = models.DateTimeField(default=timezone.now, verbose_name='上传时间', db_index=True)
    parsed_count = models.IntegerField(verbose_name='解析记录数', default=0)
    parse_status = models.CharField(max_length=20, default='pending', verbose_name='解析状态')
    error_message = models.TextField(verbose_name='错误信息', null=True, blank=True)
    
    class Meta:
        db_table = 'data_uploaded_file'
        verbose_name = '上传文件'
        verbose_name_plural = '上传文件'
        ordering = ['-upload_time']
    
    def __str__(self):
        return f'{self.filename} ({self.get_data_type_display()})'


class MicroseismicData(models.Model):
    """微震数据模型（用于数据查看）"""
    # 基础字段
    timestamp = models.DateTimeField(verbose_name='时间戳', db_index=True)
    event_id = models.CharField(max_length=50, verbose_name='事件ID', null=True, blank=True)
    
    # 坐标字段
    event_x = models.FloatField(verbose_name='X坐标')
    event_y = models.FloatField(verbose_name='Y坐标')
    event_z = models.FloatField(verbose_name='Z坐标', null=True, blank=True)
    
    # 能量和震级
    energy = models.FloatField(verbose_name='能量', null=True, blank=True)
    magnitude = models.FloatField(verbose_name='震级', null=True, blank=True)
    locate_mw = models.FloatField(verbose_name='Locate_Mw', null=True, blank=True)
    
    # 额外参数
    locate_err = models.FloatField(verbose_name='定位误差', null=True, blank=True)
    velocity = models.FloatField(verbose_name='速度', null=True, blank=True)
    
    # 来源标识
    source_file = models.CharField(max_length=255, verbose_name='来源文件', null=True, blank=True)
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, null=True, blank=True, verbose_name='上传文件')
    is_simulated = models.BooleanField(default=False, verbose_name='是否为模拟数据', db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='导入时间')
    
    class Meta:
        db_table = 'data_microseismic'
        verbose_name = '微震数据'
        verbose_name_plural = '微震数据'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['source_file']),
            models.Index(fields=['is_simulated', 'timestamp']),
            models.Index(fields=['event_id']),
        ]
    
    def __str__(self):
        event_desc = self.event_id or self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        mag = self.magnitude or self.locate_mw or 'N/A'
        return f'{event_desc} - 震级{mag}'


class SupportResistanceData(models.Model):
    """支架阻力数据模型（用于数据查看）"""
    timestamp = models.DateTimeField(verbose_name='时间戳', db_index=True)
    station_id = models.CharField(max_length=50, verbose_name='测站ID', db_index=True)
    resistance = models.FloatField(verbose_name='阻力值(MPa)')
    pressure_level = models.CharField(max_length=20, verbose_name='压力等级', null=True, blank=True)
    source_file = models.CharField(max_length=255, verbose_name='来源文件', null=True, blank=True)
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, null=True, blank=True, verbose_name='上传文件')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='导入时间')
    
    class Meta:
        db_table = 'data_support_resistance'
        verbose_name = '支架阻力数据'
        verbose_name_plural = '支架阻力数据'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp', 'station_id']),
            models.Index(fields=['source_file']),
        ]
    
    def __str__(self):
        return f'{self.station_id} - {self.timestamp}: {self.resistance}MPa'
