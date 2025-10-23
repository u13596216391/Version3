from rest_framework import serializers
from .models import SystemStatistics, AlertRecord


class SystemStatisticsSerializer(serializers.ModelSerializer):
    """系统统计数据序列化器"""
    
    class Meta:
        model = SystemStatistics
        fields = '__all__'


class AlertRecordSerializer(serializers.ModelSerializer):
    """预警记录序列化器"""
    
    class Meta:
        model = AlertRecord
        fields = '__all__'
