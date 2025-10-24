# dashboard_app/health.py
# 健康检查视图

from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
import sys

def health_check(request):
    """
    健康检查端点
    检查数据库和缓存连接状态
    """
    status = {
        'status': 'healthy',
        'checks': {}
    }
    
    # 检查数据库
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
        status['checks']['database'] = 'ok'
    except Exception as e:
        status['checks']['database'] = f'error: {str(e)}'
        status['status'] = 'unhealthy'
    
    # 检查 Redis 缓存
    try:
        cache.set('health_check', 'ok', 10)
        result = cache.get('health_check')
        status['checks']['cache'] = 'ok' if result == 'ok' else 'error'
    except Exception as e:
        status['checks']['cache'] = f'error: {str(e)}'
        status['status'] = 'unhealthy'
    
    # Python 版本
    status['python_version'] = sys.version.split()[0]
    
    return JsonResponse(status, status=200 if status['status'] == 'healthy' else 503)
