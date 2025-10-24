from rest_framework import serializers
from .models import MicroseismicData, SupportResistanceData, UploadedFile


class MicroseismicDataSerializer(serializers.ModelSerializer):
    """微震数据序列化器"""
    class Meta:
        model = MicroseismicData
        fields = ['id', 'timestamp', 'event_id', 'event_x', 'event_y', 'event_z', 
                  'energy', 'magnitude', 'locate_mw', 'locate_err', 'velocity',
                  'source_file', 'is_simulated', 'created_at']


class SupportResistanceDataSerializer(serializers.ModelSerializer):
    """支架阻力数据序列化器"""
    class Meta:
        model = SupportResistanceData
        fields = ['id', 'timestamp', 'station_id', 'resistance', 'pressure_level', 'source_file', 'created_at']


class UploadedFileSerializer(serializers.ModelSerializer):
    """上传文件序列化器"""
    file_type_display = serializers.CharField(source='get_file_type_display', read_only=True)
    data_type_display = serializers.CharField(source='get_data_type_display', read_only=True)
    
    class Meta:
        model = UploadedFile
        fields = ['id', 'filename', 'file_type', 'file_type_display', 'data_type', 'data_type_display', 
                  'file_size', 'upload_time', 'parsed_count', 'parse_status', 'error_message']
