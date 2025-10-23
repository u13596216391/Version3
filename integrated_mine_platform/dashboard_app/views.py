from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Count, Q
from datetime import timedelta
from django.utils import timezone
from .models import SystemStatistics, AlertRecord
from .serializers import SystemStatisticsSerializer, AlertRecordSerializer


class DashboardOverviewView(APIView):
    """数据大屏总览API"""
    
    def get(self, request):
        """获取大屏总览数据"""
        
        # 获取最新的统计数据
        latest_stats = SystemStatistics.objects.first()
        
        if not latest_stats:
            # 如果没有统计数据，创建初始数据
            latest_stats = SystemStatistics.objects.create()
        
        # 获取最近24小时的预警
        last_24h = timezone.now() - timedelta(hours=24)
        recent_alerts = AlertRecord.objects.filter(
            timestamp__gte=last_24h
        ).order_by('-timestamp')[:10]
        
        # 统计各级别预警数量
        alert_stats = AlertRecord.objects.filter(
            timestamp__gte=last_24h
        ).values('level').annotate(count=Count('id'))
        
        # 获取过去7天的任务趋势
        past_7_days = timezone.now() - timedelta(days=7)
        task_trend = SystemStatistics.objects.filter(
            timestamp__gte=past_7_days
        ).values('timestamp', 'completed_tasks', 'failed_tasks')[:24]
        
        response_data = {
            'overview': SystemStatisticsSerializer(latest_stats).data,
            'recent_alerts': AlertRecordSerializer(recent_alerts, many=True).data,
            'alert_stats': list(alert_stats),
            'task_trend': list(task_trend),
            'timestamp': timezone.now().isoformat()
        }
        
        return Response(response_data, status=status.HTTP_200_OK)


class StatisticsUpdateView(APIView):
    """更新统计数据API"""
    
    def post(self, request):
        """更新统计数据"""
        serializer = SystemStatisticsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AlertListView(APIView):
    """预警列表API"""
    
    def get(self, request):
        """获取预警列表"""
        level = request.query_params.get('level', None)
        limit = int(request.query_params.get('limit', 50))
        
        queryset = AlertRecord.objects.all()
        
        if level:
            queryset = queryset.filter(level=level)
        
        alerts = queryset[:limit]
        serializer = AlertRecordSerializer(alerts, many=True)
        
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        """创建新预警"""
        serializer = AlertRecordSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DataViewAPIView(APIView):
    """数据库数据查看API"""
    
    def get(self, request):
        """获取所有数据库表和记录概览"""
        from django.apps import apps
        
        data = {}
        
        # 遍历所有已安装的应用
        for app_config in apps.get_app_configs():
            if app_config.name in ['predictor_app', 'microseismic_app', 'monitoring_app', 'dashboard_app']:
                app_data = {}
                for model in app_config.get_models():
                    model_name = model.__name__
                    count = model.objects.count()
                    app_data[model_name] = {
                        'count': count,
                        'verbose_name': model._meta.verbose_name
                    }
                data[app_config.name] = app_data
        
        return Response(data, status=status.HTTP_200_OK)
