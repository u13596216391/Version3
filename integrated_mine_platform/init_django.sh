#!/bin/bash
# Django 初始化并启动脚本 - 仅用于backend服务

set -e

echo "等待数据库准备就绪..."
python << END
import sys
import time
import os
from django.db import connections
from django.db.utils import OperationalError

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        db_conn = connections['default']
        db_conn.cursor()
        print("数据库连接成功！")
        break
    except OperationalError:
        retry_count += 1
        print(f"数据库未准备好，重试 {retry_count}/{max_retries}...")
        time.sleep(2)

if retry_count >= max_retries:
    print("无法连接到数据库！")
    sys.exit(1)
END

echo "执行数据库迁移..."
python manage.py makemigrations --noinput || true
python manage.py migrate --noinput

echo "收集静态文件..."
python manage.py collectstatic --noinput --clear

echo "创建超级用户（如果不存在）..."
python manage.py shell << END || true
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123456')
    print('超级用户已创建: admin/admin123456')
else:
    print('超级用户已存在')
END

echo "初始化监控数据..."
python manage.py init_monitoring_data --initial-count 20

echo "启动 Django 服务..."
# 生产环境建议使用 gunicorn
if [ "$DJANGO_ENV" = "production" ]; then
    echo "以生产模式启动（使用 gunicorn）..."
    gunicorn integrated_mine.wsgi:application \
        --bind 0.0.0.0:8000 \
        --workers 4 \
        --threads 2 \
        --timeout 120 \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --log-level info
else
    echo "以开发模式启动..."
    python manage.py runserver 0.0.0.0:8000
fi
