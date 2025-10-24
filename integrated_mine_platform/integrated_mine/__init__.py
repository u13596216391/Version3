# Integrated Mine Platform - 集成矿山智能预测平台

# 导入Celery应用以确保在Django启动时加载
from .celery import app as celery_app

__all__ = ('celery_app',)
