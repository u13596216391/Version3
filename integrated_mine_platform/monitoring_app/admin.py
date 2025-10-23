from django.contrib import admin
from .models import MonitoringData


@admin.register(MonitoringData)
class MonitoringDataAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'data_type', 'location', 'value', 'unit', 'is_abnormal']
    list_filter = ['data_type', 'is_abnormal', 'timestamp', 'location']
    search_fields = ['location']
    ordering = ['-timestamp']
