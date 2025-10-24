from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.utils import timezone
from datetime import timedelta
from .services import (
    get_microseismic_analysis,
    get_support_dwt_analysis,
    get_wavelet_comparison,
)
from .models import AnalysisResult


class MicroseismicScatterView(APIView):
    """微震散点图分析API"""
    
    def get(self, request):
        """
        GET /api/analysis/microseismic/scatter/
        参数:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            analysis_type: 分析类型 (frequency/energy), 默认frequency
            dataset_id: 数据集ID（可选，如果指定则分析上传的数据集）
                       支持: 'simulated'(模拟数据), source_file名称, 或uploaded_file的ID
        """
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        analysis_type = request.query_params.get('analysis_type', 'frequency')
        dataset_id = request.query_params.get('dataset_id')
        
        if not all([start_date, end_date]):
            return Response({
                'success': False,
                'error': 'start_date和end_date参数是必需的'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # 定义辅助线(工作面、巷道等) - 参照实际矿井布局
            # 红色：x=1750，y范围0~300
            # 黄色：y=300，x范围1750~0
            # 紫色：y=0，x范围0~300
            auxiliary_lines = [
                {
                    "coords": [[1750, 0], [1750, 300]],
                    "color": "red",
                    "name": "Working Face Boundary"
                },
                {
                    "coords": [[1750, 300], [0, 300]],
                    "color": "yellow",
                    "name": "Upper Roadway"
                },
                {
                    "coords": [[0, 0], [300, 0]],
                    "color": "purple",
                    "name": "Lower Roadway"
                }
            ]
            
            result = get_microseismic_analysis(
                start_date=start_date,
                end_date=end_date,
                analysis_type=analysis_type,
                auxiliary_lines=auxiliary_lines,
                dataset_id=dataset_id
            )
            
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"散点图分析错误: {error_detail}")
            return Response({
                'success': False,
                'error': str(e),
                'detail': error_detail
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MicroseismicDensityView(APIView):
    """微震核密度图分析API"""
    
    def get(self, request):
        """
        GET /api/analysis/microseismic/density/
        参数:
            start_date: 开始日期
            end_date: 结束日期
            analysis_type: 分析类型 (frequency/energy), 默认frequency
            dataset_id: 数据集ID（可选，如果指定则分析上传的数据集）
        """
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        analysis_type = request.query_params.get('analysis_type', 'frequency')
        dataset_id = request.query_params.get('dataset_id')
        
        if not all([start_date, end_date]):
            return Response({
                'success': False,
                'error': 'start_date和end_date参数是必需的'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # 定义辅助线(工作面、巷道等) - 参照实际矿井布局
            # 红色：x=1750，y范围0~300
            # 黄色：y=300，x范围1750~0
            # 紫色：y=0，x范围0~300
            auxiliary_lines = [
                {
                    "coords": [[1750, 0], [1750, 300]],
                    "color": "red",
                    "name": "Working Face Boundary"
                },
                {
                    "coords": [[1750, 300], [0, 300]],
                    "color": "yellow",
                    "name": "Upper Roadway"
                },
                {
                    "coords": [[0, 0], [300, 0]],
                    "color": "purple",
                    "name": "Lower Roadway"
                }
            ]
            
            result = get_microseismic_analysis(
                start_date=start_date,
                end_date=end_date,
                analysis_type=analysis_type,
                auxiliary_lines=auxiliary_lines,
                dataset_id=dataset_id
            )
            
            # 缓存结果
            if result.get('success'):
                AnalysisResult.objects.create(
                    analysis_type='microseismic_density',
                    parameters={
                        'start_date': start_date,
                        'end_date': end_date,
                        'analysis_type': analysis_type,
                        'dataset_id': dataset_id
                    },
                    result_data=result,
                    image_data=result.get('density_plot'),
                    expires_at=timezone.now() + timedelta(hours=24)
                )
            
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"密度图分析错误: {error_detail}")
            return Response({
                'success': False,
                'error': str(e),
                'detail': error_detail
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SupportDWTAnalysisView(APIView):
    """支架阻力DWT分析API"""
    
    def get(self, request):
        """
        GET /api/analysis/support/dwt/
        参数:
            station_id: 测站ID
            start_date: 开始日期
            end_date: 结束日期
            wavelet: 小波基函数 (默认db4)
            dataset_id: 数据集ID（可选，如果指定则分析上传的数据集）
        """
        station_id = request.query_params.get('station_id')
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        wavelet = request.query_params.get('wavelet', 'db4')
        dataset_id = request.query_params.get('dataset_id')
        
        if not all([station_id, start_date, end_date]):
            return Response({
                'success': False,
                'error': 'station_id, start_date和end_date参数是必需的'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = get_support_dwt_analysis(
                station_id=station_id,
                start_date=start_date,
                end_date=end_date,
                wavelet=wavelet,
                dataset_id=dataset_id
            )
            
            # 缓存结果
            if result.get('success'):
                AnalysisResult.objects.create(
                    analysis_type='support_dwt',
                    parameters={
                        'station_id': station_id,
                        'start_date': start_date,
                        'end_date': end_date,
                        'wavelet': wavelet,
                        'dataset_id': dataset_id
                    },
                    result_data=result,
                    image_data=result.get('dwt_plot'),
                    expires_at=timezone.now() + timedelta(hours=24)
                )
            
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SupportWaveletComparisonView(APIView):
    """支架阻力小波对比分析API"""
    
    def get(self, request):
        """
        GET /api/analysis/support/wavelet-comparison/
        参数:
            station_id: 测站ID
            start_date: 开始日期
            end_date: 结束日期
            wavelets[]: 小波基函数列表 (可多选)
            dataset_id: 数据集ID（可选，如果指定则分析上传的数据集）
        """
        station_id = request.query_params.get('station_id')
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        wavelets = request.query_params.getlist('wavelets[]')
        dataset_id = request.query_params.get('dataset_id')
        
        if not wavelets:
            wavelets = ['db4', 'sym4', 'coif4']  # 默认对比这三种
        
        if not all([station_id, start_date, end_date]):
            return Response({
                'success': False,
                'error': 'station_id, start_date和end_date参数是必需的'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = get_wavelet_comparison(
                station_id=station_id,
                start_date=start_date,
                end_date=end_date,
                wavelets=wavelets,
                dataset_id=dataset_id
            )
            
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_available_stations(request):
    """获取可用的测站列表"""
    from .models import SupportResistance
    from django.db.models import Count
    
    stations = SupportResistance.objects.values('station_id').annotate(
        count=Count('id')
    ).order_by('-count')
    
    return Response({
        'success': True,
        'stations': list(stations)
    })


@api_view(['GET'])
def get_analysis_history(request):
    """获取分析历史记录"""
    analysis_type = request.query_params.get('analysis_type')
    limit = int(request.query_params.get('limit', 10))
    
    query = AnalysisResult.objects.all()
    if analysis_type:
        query = query.filter(analysis_type=analysis_type)
    
    results = query.order_by('-created_at')[:limit]
    
    data = []
    for result in results:
        if not result.is_expired():
            data.append({
                'id': result.id,
                'analysis_type': result.analysis_type,
                'parameters': result.parameters,
                'created_at': result.created_at.isoformat(),
                'has_image': bool(result.image_data)
            })
    
    return Response({
        'success': True,
        'count': len(data),
        'results': data
    })
