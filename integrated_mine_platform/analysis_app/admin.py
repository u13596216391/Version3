from django.contrib import admin
from .models import MicroseismicEvent, SupportResistance, ProgressData, AnalysisResult


@admin.register(MicroseismicEvent)
class MicroseismicEventAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'x_coord', 'y_coord', 'z_coord', 'energy', 'magnitude', 'data_type', 'is_simulated')
    list_filter = ('data_type', 'is_simulated', 'timestamp')
    search_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    ordering = ('-timestamp',)


@admin.register(SupportResistance)
class SupportResistanceAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'station_id', 'resistance_value', 'pressure_level', 'is_abnormal', 'is_simulated')
    list_filter = ('station_id', 'is_abnormal', 'is_simulated', 'timestamp')
    search_fields = ('station_id',)
    date_hierarchy = 'timestamp'
    ordering = ('-timestamp',)


@admin.register(ProgressData)
class ProgressDataAdmin(admin.ModelAdmin):
    list_display = ('date', 'work_face', 'progress', 'notes')
    list_filter = ('work_face', 'date')
    search_fields = ('work_face', 'notes')
    date_hierarchy = 'date'
    ordering = ('-date',)


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('analysis_type', 'created_at', 'is_expired', 'get_parameters_summary')
    list_filter = ('analysis_type', 'created_at')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)
    
    def get_parameters_summary(self, obj):
        """显示参数摘要"""
        params = obj.parameters
        if 'start_date' in params and 'end_date' in params:
            return f"{params['start_date']} ~ {params['end_date']}"
        return str(params)[:50]
    get_parameters_summary.short_description = '参数摘要'
