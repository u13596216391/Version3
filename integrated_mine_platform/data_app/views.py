from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.paginator import Paginator
from django.db.models import Q, Count, Min, Max
from django.utils import timezone
from datetime import datetime
from .models import MicroseismicData, SupportResistanceData, UploadedFile
from .serializers import MicroseismicDataSerializer, SupportResistanceDataSerializer, UploadedFileSerializer
from .parsers import process_uploaded_file


class FileUploadView(APIView):
    """文件上传API"""
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        """
        上传CSV或ZIP文件
        
        参数:
            file: 文件对象
            data_type: 数据类型 (microseismic/support_resistance)
        """
        file_obj = request.FILES.get('file')
        data_type = request.data.get('data_type')
        
        if not file_obj:
            return Response({
                'success': False,
                'error': '请选择要上传的文件'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if data_type not in ['microseismic', 'support_resistance']:
            return Response({
                'success': False,
                'error': '无效的数据类型'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        filename = file_obj.name
        file_size = file_obj.size
        
        # 检查文件类型
        if filename.lower().endswith('.csv'):
            file_type = 'csv'
        elif filename.lower().endswith('.zip'):
            file_type = 'zip'
        else:
            return Response({
                'success': False,
                'error': '仅支持CSV和ZIP文件'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 创建上传记录
        uploaded_file = UploadedFile.objects.create(
            filename=filename,
            file_type=file_type,
            data_type=data_type,
            file_size=file_size,
            parse_status='parsing'
        )
        
        try:
            # 处理文件
            success, count, error = process_uploaded_file(uploaded_file, data_type, file_obj)
            
            if success:
                uploaded_file.parse_status = 'success'
                uploaded_file.parsed_count = count
                uploaded_file.save()
                
                return Response({
                    'success': True,
                    'message': f'文件上传成功，解析了 {count} 条数据',
                    'uploaded_file': UploadedFileSerializer(uploaded_file).data
                }, status=status.HTTP_201_CREATED)
            else:
                uploaded_file.parse_status = 'failed'
                uploaded_file.error_message = error
                uploaded_file.save()
                
                return Response({
                    'success': False,
                    'error': error,
                    'uploaded_file': UploadedFileSerializer(uploaded_file).data
                }, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            uploaded_file.parse_status = 'failed'
            uploaded_file.error_message = str(e)
            uploaded_file.save()
            
            return Response({
                'success': False,
                'error': f'文件处理失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MicroseismicDataListView(APIView):
    """微震数据列表API"""
    
    def get(self, request):
        """
        获取微震数据列表
        
        参数:
            page: 页码 (默认1)
            page_size: 每页数量 (默认50)
            start_date: 开始日期
            end_date: 结束日期
            source_file: 源文件名
            min_x, max_x: X坐标范围
            min_y, max_y: Y坐标范围
        """
        # 获取查询参数
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 50))
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        source_file = request.query_params.get('source_file')
        min_x = request.query_params.get('min_x')
        max_x = request.query_params.get('max_x')
        min_y = request.query_params.get('min_y')
        max_y = request.query_params.get('max_y')
        
        # 构建查询
        queryset = MicroseismicData.objects.all()
        
        if start_date:
            queryset = queryset.filter(timestamp__gte=start_date)
        
        if end_date:
            queryset = queryset.filter(timestamp__lte=end_date)
        
        if source_file:
            queryset = queryset.filter(source_file__icontains=source_file)
        
        if min_x:
            queryset = queryset.filter(event_x__gte=float(min_x))
        
        if max_x:
            queryset = queryset.filter(event_x__lte=float(max_x))
        
        if min_y:
            queryset = queryset.filter(event_y__gte=float(min_y))
        
        if max_y:
            queryset = queryset.filter(event_y__lte=float(max_y))
        
        # 统计信息
        total_count = queryset.count()
        
        # 分页
        paginator = Paginator(queryset, page_size)
        page_obj = paginator.get_page(page)
        
        serializer = MicroseismicDataSerializer(page_obj.object_list, many=True)
        
        return Response({
            'success': True,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': paginator.num_pages,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'data': serializer.data
        }, status=status.HTTP_200_OK)


class SupportResistanceDataListView(APIView):
    """支架阻力数据列表API"""
    
    def get(self, request):
        """
        获取支架阻力数据列表
        
        参数:
            page: 页码
            page_size: 每页数量
            start_date: 开始日期
            end_date: 结束日期
            station_id: 测站ID
            source_file: 源文件名
            min_resistance, max_resistance: 阻力范围
        """
        # 获取查询参数
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 50))
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        station_id = request.query_params.get('station_id')
        source_file = request.query_params.get('source_file')
        min_resistance = request.query_params.get('min_resistance')
        max_resistance = request.query_params.get('max_resistance')
        
        # 构建查询
        queryset = SupportResistanceData.objects.all()
        
        if start_date:
            queryset = queryset.filter(timestamp__gte=start_date)
        
        if end_date:
            queryset = queryset.filter(timestamp__lte=end_date)
        
        if station_id:
            queryset = queryset.filter(station_id__icontains=station_id)
        
        if source_file:
            queryset = queryset.filter(source_file__icontains=source_file)
        
        if min_resistance:
            queryset = queryset.filter(resistance__gte=float(min_resistance))
        
        if max_resistance:
            queryset = queryset.filter(resistance__lte=float(max_resistance))
        
        # 统计信息
        total_count = queryset.count()
        
        # 分页
        paginator = Paginator(queryset, page_size)
        page_obj = paginator.get_page(page)
        
        serializer = SupportResistanceDataSerializer(page_obj.object_list, many=True)
        
        return Response({
            'success': True,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': paginator.num_pages,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'data': serializer.data
        }, status=status.HTTP_200_OK)


class UploadedFileListView(APIView):
    """上传文件列表API"""
    
    def get(self, request):
        """获取上传文件列表"""
        files = UploadedFile.objects.all()[:100]  # 最近100个
        serializer = UploadedFileSerializer(files, many=True)
        
        return Response({
            'success': True,
            'count': files.count(),
            'files': serializer.data
        }, status=status.HTTP_200_OK)


class DataStatisticsView(APIView):
    """数据统计API"""
    
    def get(self, request):
        """获取数据统计信息"""
        microseismic_count = MicroseismicData.objects.count()
        support_resistance_count = SupportResistanceData.objects.count()
        uploaded_files_count = UploadedFile.objects.count()
        
        # 按源文件统计
        microseismic_by_file = MicroseismicData.objects.values('source_file').annotate(count=Count('id')).order_by('-count')[:10]
        support_by_file = SupportResistanceData.objects.values('source_file').annotate(count=Count('id')).order_by('-count')[:10]
        
        # 支架阻力按测站统计
        support_by_station = SupportResistanceData.objects.values('station_id').annotate(count=Count('id')).order_by('-count')[:10]
        
        return Response({
            'success': True,
            'statistics': {
                'microseismic_total': microseismic_count,
                'support_resistance_total': support_resistance_count,
                'uploaded_files_total': uploaded_files_count,
                'microseismic_by_file': list(microseismic_by_file),
                'support_by_file': list(support_by_file),
                'support_by_station': list(support_by_station),
            }
        }, status=status.HTTP_200_OK)


class DatasetListView(APIView):
    """数据集列表API - 用于分析模块选择数据集"""
    
    def get(self, request):
        """
        获取可用的数据集列表
        
        参数:
            data_type: 数据类型 (microseismic/support_resistance)，默认为 microseismic
        
        返回按上传批次划分的数据集，包括：
            - 示例数据集（合并了模拟数据）
            - 按上传批次（uploaded_file）组织的数据集
        """
        data_type = request.query_params.get('data_type', 'microseismic')
        
        if data_type not in ['microseismic', 'support_resistance']:
            return Response({
                'success': False,
                'error': '无效的数据类型，必须是 microseismic 或 support_resistance'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        datasets = []
        
        if data_type == 'microseismic':
            # 1. 添加微震示例数据集（合并所有 is_simulated=True 的数据）
            simulated_count = MicroseismicData.objects.filter(is_simulated=True).count()
            if simulated_count > 0:
                simulated_data = MicroseismicData.objects.filter(is_simulated=True).aggregate(
                    min_time=Min('timestamp'),
                    max_time=Max('timestamp')
                )
                datasets.append({
                    'id': 'simulated',
                    'name': '示例数据集',
                    'type': 'simulated',
                    'data_type': 'microseismic',
                    'count': simulated_count,
                    'time_range': {
                        'start': simulated_data['min_time'].isoformat() if simulated_data['min_time'] else None,
                        'end': simulated_data['max_time'].isoformat() if simulated_data['max_time'] else None
                    }
                })
            
            # 2. 按上传批次（uploaded_file）组织微震数据集
            uploaded_files = UploadedFile.objects.filter(
                data_type='microseismic',
                parse_status='success'
            ).order_by('-upload_time')
            
            for uf in uploaded_files:
                # 统计该上传文件的数据
                file_data = MicroseismicData.objects.filter(uploaded_file=uf).aggregate(
                    count=Count('id'),
                    min_time=Min('timestamp'),
                    max_time=Max('timestamp')
                )
                
                if file_data['count'] > 0:
                    # 格式化上传日期作为显示名称
                    upload_date = uf.upload_time.strftime('%Y年%m月%d日')
                    datasets.append({
                        'id': str(uf.id),  # 使用 uploaded_file 的 ID
                        'name': f'{upload_date}上传 - {uf.filename}',
                        'type': 'uploaded',
                        'data_type': 'microseismic',
                        'count': file_data['count'],
                        'upload_time': uf.upload_time.isoformat(),
                        'time_range': {
                            'start': file_data['min_time'].isoformat() if file_data['min_time'] else None,
                            'end': file_data['max_time'].isoformat() if file_data['max_time'] else None
                        }
                    })
        
        elif data_type == 'support_resistance':
            # 1. 添加支架阻力示例数据集（如果有 is_simulated 字段的话）
            # 注意：当前 SupportResistanceData 模型没有 is_simulated 字段，可以后续添加
            
            # 2. 按上传批次（uploaded_file）组织支架阻力数据集
            uploaded_files = UploadedFile.objects.filter(
                data_type='support_resistance',
                parse_status='success'
            ).order_by('-upload_time')
            
            for uf in uploaded_files:
                # 统计该上传文件的数据
                file_data = SupportResistanceData.objects.filter(uploaded_file=uf).aggregate(
                    count=Count('id'),
                    min_time=Min('timestamp'),
                    max_time=Max('timestamp')
                )
                
                if file_data['count'] > 0:
                    # 格式化上传日期作为显示名称
                    upload_date = uf.upload_time.strftime('%Y年%m月%d日')
                    datasets.append({
                        'id': str(uf.id),  # 使用 uploaded_file 的 ID
                        'name': f'{upload_date}上传 - {uf.filename}',
                        'type': 'uploaded',
                        'data_type': 'support_resistance',
                        'count': file_data['count'],
                        'upload_time': uf.upload_time.isoformat(),
                        'time_range': {
                            'start': file_data['min_time'].isoformat() if file_data['min_time'] else None,
                            'end': file_data['max_time'].isoformat() if file_data['max_time'] else None
                        }
                    })
        
        return Response({
            'success': True,
            'datasets': datasets,
            'total': len(datasets),
            'data_type': data_type
        }, status=status.HTTP_200_OK)


class DatasetDataView(APIView):
    """获取指定数据集的数据"""
    
    def get(self, request):
        """
        获取指定数据集的微震数据
        
        参数:
            dataset_id: 数据集ID (source_file名称或'simulated')
            limit: 返回数据条数限制 (默认1000)
        """
        dataset_id = request.GET.get('dataset_id')
        limit = int(request.GET.get('limit', 1000))
        
        if not dataset_id:
            return Response({
                'success': False,
                'error': '请指定dataset_id'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 查询数据
        if dataset_id == 'simulated':
            queryset = MicroseismicData.objects.filter(is_simulated=True)
        else:
            queryset = MicroseismicData.objects.filter(source_file=dataset_id)
        
        # 限制返回数量
        queryset = queryset[:limit]
        
        # 序列化
        serializer = MicroseismicDataSerializer(queryset, many=True)
        
        return Response({
            'success': True,
            'count': queryset.count(),
            'data': serializer.data
        }, status=status.HTTP_200_OK)


class DemoDataManagementView(APIView):
    """演示数据管理API"""
    
    def get(self, request):
        """获取演示数据统计"""
        demo_count = MicroseismicData.objects.filter(is_simulated=True).count()
        
        if demo_count > 0:
            demo_stats = MicroseismicData.objects.filter(is_simulated=True).aggregate(
                min_time=Min('timestamp'),
                max_time=Max('timestamp')
            )
            
            return Response({
                'success': True,
                'has_demo_data': True,
                'count': demo_count,
                'time_range': {
                    'start': demo_stats['min_time'],
                    'end': demo_stats['max_time']
                }
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'success': True,
                'has_demo_data': False,
                'count': 0
            }, status=status.HTTP_200_OK)
    
    def post(self, request):
        """生成演示数据"""
        import random
        import numpy as np
        from datetime import timedelta
        
        count = int(request.data.get('count', 500))
        days = int(request.data.get('days', 30))
        
        if count > 5000:
            return Response({
                'success': False,
                'error': '单次生成数量不能超过5000条'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # 时间范围
            end_time = timezone.now()
            start_time = end_time - timedelta(days=days)
            
            # 空间范围 (模拟一个矿区)
            x_range = (0, 1700)  # 工作面长度
            y_range = (-200, 200)  # 巷道宽度
            z_range = (-800, -300)  # 深度范围
            
            # 生成数据集中的"热点区域" (模拟应力集中区)
            hotspots = [
                {'center': (850, 0, -500), 'radius': 200, 'weight': 0.4},
                {'center': (300, 50, -400), 'radius': 150, 'weight': 0.3},
                {'center': (1400, -30, -600), 'radius': 180, 'weight': 0.3},
            ]
            
            records = []
            for i in range(count):
                # 生成时间戳
                time_delta = timedelta(seconds=random.randint(0, int(days * 24 * 3600)))
                timestamp = start_time + time_delta
                
                # 决定是否在热点区域 (70%概率)
                if random.random() < 0.7:
                    hotspot = random.choices(hotspots, weights=[h['weight'] for h in hotspots])[0]
                    # 在热点周围生成，使用正态分布
                    event_x = np.random.normal(hotspot['center'][0], hotspot['radius'] / 3)
                    event_y = np.random.normal(hotspot['center'][1], hotspot['radius'] / 3)
                    event_z = np.random.normal(hotspot['center'][2], hotspot['radius'] / 3)
                    # 热点区域能量更高
                    energy_base = random.uniform(1e6, 1e8)
                else:
                    # 随机分布
                    event_x = random.uniform(*x_range)
                    event_y = random.uniform(*y_range)
                    event_z = random.uniform(*z_range)
                    energy_base = random.uniform(1e4, 1e6)
                
                # 限制在范围内
                event_x = max(x_range[0], min(x_range[1], event_x))
                event_y = max(y_range[0], min(y_range[1], event_y))
                event_z = max(z_range[0], min(z_range[1], event_z))
                
                # 计算能量和震级 (使用对数关系)
                energy = energy_base * random.uniform(0.5, 2.0)
                magnitude = (np.log10(energy) - 4.8) / 1.5  # 里氏震级公式简化版
                
                # 生成事件ID
                event_id = f'DEMO_{timestamp.strftime("%Y%m%d")}_{i:04d}'
                
                record = MicroseismicData(
                    timestamp=timestamp,
                    event_id=event_id,
                    event_x=round(event_x, 2),
                    event_y=round(event_y, 2),
                    event_z=round(event_z, 2),
                    energy=energy,
                    magnitude=round(magnitude, 2),
                    locate_mw=round(magnitude + random.uniform(-0.2, 0.2), 2),
                    locate_err=round(random.uniform(1, 10), 2),
                    velocity=round(random.uniform(3000, 6000), 1),
                    source_file='演示数据',
                    is_simulated=True
                )
                records.append(record)
            
            # 批量创建
            MicroseismicData.objects.bulk_create(records, batch_size=500)
            
            return Response({
                'success': True,
                'message': f'成功生成{count}条演示数据',
                'count': count,
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                }
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({
                'success': False,
                'error': f'生成失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request):
        """删除所有演示数据"""
        try:
            deleted_count = MicroseismicData.objects.filter(is_simulated=True).delete()[0]
            
            return Response({
                'success': True,
                'message': f'已删除{deleted_count}条演示数据'
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'success': False,
                'error': f'删除失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StationListView(APIView):
    """获取数据集中的测站ID列表"""
    
    def get(self, request):
        """
        获取指定数据集中的所有测站ID
        
        参数:
            dataset_id: 数据集ID（必需）
        
        返回:
            stations: 测站ID列表，包含每个测站的数据统计
        """
        dataset_id = request.query_params.get('dataset_id')
        
        if not dataset_id:
            return Response({
                'success': False,
                'error': '请指定dataset_id参数'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # 根据 dataset_id 查询支架阻力数据
            if dataset_id.isdigit():
                # 上传的数据集 (uploaded_file ID)
                queryset = SupportResistanceData.objects.filter(
                    uploaded_file_id=int(dataset_id)
                )
            else:
                # source_file 数据集
                queryset = SupportResistanceData.objects.filter(
                    source_file=dataset_id
                )
            
            # 按测站ID分组统计
            from django.db.models import Count, Min, Max, Avg
            station_stats = queryset.values('station_id').annotate(
                count=Count('id'),
                min_time=Min('timestamp'),
                max_time=Max('timestamp'),
                avg_resistance=Avg('resistance')
            ).order_by('station_id')
            
            stations = []
            for stat in station_stats:
                stations.append({
                    'station_id': stat['station_id'],
                    'count': stat['count'],
                    'time_range': {
                        'start': stat['min_time'].isoformat() if stat['min_time'] else None,
                        'end': stat['max_time'].isoformat() if stat['max_time'] else None
                    },
                    'avg_resistance': round(stat['avg_resistance'], 2) if stat['avg_resistance'] else None
                })
            
            return Response({
                'success': True,
                'stations': stations,
                'total': len(stations)
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({
                'success': False,
                'error': f'查询失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


