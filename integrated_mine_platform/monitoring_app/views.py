from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.utils import timezone
from datetime import timedelta
from .models import MonitoringData
from .simulator import get_simulator
from .tasks import generate_simulated_data


class RealtimeDataView(APIView):
    """实时数据API"""
    
    def get(self, request):
        """获取最新监控数据"""
        data_type = request.query_params.get('type', None)
        location = request.query_params.get('location', None)
        hours = int(request.query_params.get('hours', 1))
        is_simulated = request.query_params.get('is_simulated', None)
        
        # 获取指定时间范围的数据
        time_threshold = timezone.now() - timedelta(hours=hours)
        queryset = MonitoringData.objects.filter(timestamp__gte=time_threshold)
        
        if data_type:
            queryset = queryset.filter(data_type=data_type)
        
        if location:
            queryset = queryset.filter(location=location)
        
        if is_simulated is not None:
            queryset = queryset.filter(is_simulated=is_simulated.lower() == 'true')
        
        data = list(queryset.values())
        
        return Response({
            'count': len(data),
            'data': data,
            'timestamp': timezone.now().isoformat()
        }, status=status.HTTP_200_OK)
    
    def post(self, request):
        """接收监控数据"""
        data = request.data
        
        try:
            monitoring_data = MonitoringData.objects.create(
                data_type=data.get('data_type'),
                location=data.get('location'),
                value=data.get('value'),
                unit=data.get('unit', ''),
                is_abnormal=data.get('is_abnormal', False),
                is_simulated=data.get('is_simulated', False)
            )
            
            return Response({
                'message': '数据接收成功',
                'id': monitoring_data.id
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


class SimulatorView(APIView):
    """模拟数据生成API"""
    
    def post(self, request):
        """手动触发生成模拟数据"""
        count = request.data.get('count', 5)
        
        try:
            # 异步执行任务
            task = generate_simulated_data.delay(count=count)
            
            return Response({
                'message': '模拟数据生成任务已启动',
                'task_id': task.id,
                'count': count
            }, status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def get(self, request):
        """获取模拟器状态和配置"""
        simulator = get_simulator()
        
        return Response({
            'data_types': list(simulator.DATA_RANGES.keys()),
            'locations': simulator.LOCATIONS,
            'ranges': simulator.DATA_RANGES,
        }, status=status.HTTP_200_OK)


@api_view(['POST'])
def generate_simulator_data_now(request):
    """立即生成模拟数据（同步）"""
    count = request.data.get('count', 5)
    
    try:
        simulator = get_simulator()
        data_list = simulator.generate_batch_data(count=count)
        
        # 批量创建数据库记录
        monitoring_objects = [
            MonitoringData(**data) for data in data_list
        ]
        created = MonitoringData.objects.bulk_create(monitoring_objects)
        
        return Response({
            'success': True,
            'message': f'成功生成 {len(created)} 条模拟数据',
            'count': len(created),
            'data': [
                {
                    'id': obj.id,
                    'data_type': obj.data_type,
                    'location': obj.location,
                    'value': obj.value,
                    'unit': obj.unit,
                    'is_abnormal': obj.is_abnormal,
                    'is_simulated': obj.is_simulated,
                    'timestamp': obj.timestamp.isoformat()
                }
                for obj in created[:10]  # 返回前10条
            ]
        }, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def dashboard_data(request):
    """获取大屏展示数据"""
    try:
        # 获取最近1小时的数据
        time_threshold = timezone.now() - timedelta(hours=1)
        recent_data_queryset = MonitoringData.objects.filter(
            timestamp__gte=time_threshold
        ).order_by('-timestamp')
        
        # 统计数据 (在切片前统计)
        total_count = recent_data_queryset.count()
        simulated_count = recent_data_queryset.filter(is_simulated=True).count()
        abnormal_count = recent_data_queryset.filter(is_abnormal=True).count()
        
        # 按类型统计
        type_stats = {}
        for data_type, label in MonitoringData.DATA_TYPES:
            type_data = recent_data_queryset.filter(data_type=data_type)
            if type_data.exists():
                values = list(type_data.values_list('value', flat=True))
                type_stats[data_type] = {
                    'label': label,
                    'avg': round(sum(values) / len(values), 2) if values else 0,
                    'max': round(max(values), 2) if values else 0,
                    'min': round(min(values), 2) if values else 0,
                    'count': len(values),
                }
        
        # 最近数据列表 (切片获取前20条)
        recent_data_list = recent_data_queryset[:20]
        recent_list = [
            {
                'id': d.id,
                'timestamp': d.timestamp.isoformat(),
                'data_type': d.data_type,
                'location': d.location,
                'value': d.value,
                'unit': d.unit,
                'is_abnormal': d.is_abnormal,
                'is_simulated': d.is_simulated,
            }
            for d in recent_data_list
        ]
        
        return Response({
            'total_count': total_count,
            'simulated_count': simulated_count,
            'real_count': total_count - simulated_count,
            'abnormal_count': abnormal_count,
            'normal_count': total_count - abnormal_count,
            'type_stats': type_stats,
            'recent_data': recent_list,
            'timestamp': timezone.now().isoformat()
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
