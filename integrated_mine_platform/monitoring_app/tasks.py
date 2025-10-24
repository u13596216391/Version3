"""
定时任务 - 自动生成模拟数据
"""
from celery import shared_task
from django.utils import timezone
from .models import MonitoringData
from .simulator import get_simulator
import logging

logger = logging.getLogger(__name__)


@shared_task
def generate_simulated_data(count=5):
    """
    生成模拟监控数据
    
    Args:
        count: 每种数据类型生成的数量
    """
    try:
        logger.info(f"DEBUG - 接收到的count参数: {count}")
        simulator = get_simulator()
        data_list = simulator.generate_batch_data(count=count)
        
        # 批量创建数据库记录
        monitoring_objects = [
            MonitoringData(**data) for data in data_list
        ]
        created = MonitoringData.objects.bulk_create(monitoring_objects)
        
        logger.info(f"成功生成 {len(created)} 条模拟数据")
        return {
            'success': True,
            'count': len(created),
            'timestamp': timezone.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"生成模拟数据失败: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }


@shared_task
def cleanup_old_simulated_data(days=7):
    """
    清理旧的模拟数据
    
    Args:
        days: 保留天数
    """
    try:
        cutoff_date = timezone.now() - timezone.timedelta(days=days)
        deleted_count, _ = MonitoringData.objects.filter(
            is_simulated=True,
            timestamp__lt=cutoff_date
        ).delete()
        
        logger.info(f"清理了 {deleted_count} 条旧模拟数据")
        return {
            'success': True,
            'deleted_count': deleted_count,
            'timestamp': timezone.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"清理模拟数据失败: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }
