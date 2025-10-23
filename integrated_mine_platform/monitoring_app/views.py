from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import timedelta
from .models import MonitoringData


class RealtimeDataView(APIView):
    """实时数据API"""
    
    def get(self, request):
        """获取最新监控数据"""
        data_type = request.query_params.get('type', None)
        location = request.query_params.get('location', None)
        hours = int(request.query_params.get('hours', 1))
        
        # 获取指定时间范围的数据
        time_threshold = timezone.now() - timedelta(hours=hours)
        queryset = MonitoringData.objects.filter(timestamp__gte=time_threshold)
        
        if data_type:
            queryset = queryset.filter(data_type=data_type)
        
        if location:
            queryset = queryset.filter(location=location)
        
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
                is_abnormal=data.get('is_abnormal', False)
            )
            
            return Response({
                'message': '数据接收成功',
                'id': monitoring_data.id
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
