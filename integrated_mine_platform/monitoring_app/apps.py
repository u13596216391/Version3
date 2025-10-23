from django.apps import AppConfig


class MonitoringAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monitoring_app'
    verbose_name = '实时监控'
