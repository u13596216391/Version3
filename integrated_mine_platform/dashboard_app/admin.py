from django.contrib import admin
from .models import SystemStatistics, AlertRecord


@admin.register(SystemStatistics)
class SystemStatisticsAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'total_tasks', 'running_tasks', 'completed_tasks', 'system_health']
    list_filter = ['system_health', 'timestamp']
    ordering = ['-timestamp']


@admin.register(AlertRecord)
class AlertRecordAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'level', 'title', 'source', 'is_resolved']
    list_filter = ['level', 'is_resolved', 'timestamp']
    search_fields = ['title', 'message', 'source']
    ordering = ['-timestamp']
