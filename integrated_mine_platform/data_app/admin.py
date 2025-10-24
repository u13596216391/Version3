from django.contrib import admin
from .models import MicroseismicData, SupportResistanceData, UploadedFile


@admin.register(MicroseismicData)
class MicroseismicDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'timestamp', 'event_x', 'event_y', 'event_z', 'energy', 'magnitude', 'source_file', 'created_at']
    list_filter = ['source_file', 'created_at']
    search_fields = ['source_file']
    date_hierarchy = 'timestamp'


@admin.register(SupportResistanceData)
class SupportResistanceDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'timestamp', 'station_id', 'resistance', 'pressure_level', 'source_file', 'created_at']
    list_filter = ['station_id', 'source_file', 'created_at']
    search_fields = ['station_id', 'source_file']
    date_hierarchy = 'timestamp'


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['id', 'filename', 'file_type', 'data_type', 'file_size', 'upload_time', 'parsed_count', 'parse_status']
    list_filter = ['file_type', 'data_type', 'parse_status', 'upload_time']
    search_fields = ['filename']
    readonly_fields = ['upload_time', 'parsed_count', 'parse_status', 'error_message']
