"""
Celery配置文件
"""
import os
from celery import Celery

# 设置默认Django settings模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'integrated_mine.settings')

app = Celery('integrated_mine')

# 从Django配置中加载Celery配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现任务
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    """调试任务"""
    print(f'Request: {self.request!r}')
