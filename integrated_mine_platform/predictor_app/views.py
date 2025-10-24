# predictor_app/views.py

import os
import tempfile
import json
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .models import TrainingRun, ModelResult
from .services import start_training_session_async

@method_decorator(csrf_exempt, name='dispatch')
class StartTrainingView(APIView):
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    
    def post(self, request, *args, **kwargs):
        # 添加调试日志
        print(f"DEBUG - request.data: {dict(request.data)}")
        print(f"DEBUG - request.FILES: {dict(request.FILES)}")
        print(f"DEBUG - request.POST: {dict(request.POST)}")
        
        config_str = request.data.get('config')
        print(f"DEBUG - config_str: {config_str}")
        
        if not config_str:
            print("ERROR - 缺少config配置")
            return Response({"error": "缺少config配置。"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            return Response({"error": "Config必须是有效的JSON字符串。"}, status=status.HTTP_400_BAD_REQUEST)
        
        # 支持两种数据源：文件上传或数据集选择
        file = request.FILES.get('file')
        dataset_id = request.data.get('dataset_id')
        
        print(f"DEBUG - file: {file}")
        print(f"DEBUG - dataset_id: {dataset_id}")
        print(f"DEBUG - dataset_id type: {type(dataset_id)}")
        print(f"DEBUG - dataset_id bool: {bool(dataset_id)}")
        
        if file:
            # 文件上传方式
            print("INFO - 使用文件上传模式")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
        elif dataset_id and dataset_id.strip():  # 检查不为空且不是空字符串
            # 数据集选择方式
            print(f"INFO - 使用数据集模式, ID: {dataset_id}")
            from data_app.models import UploadedFile, MicroseismicData, SupportResistanceData
            
            try:
                # 特殊处理：示例数据集
                if dataset_id == 'simulated':
                    print("INFO - 使用示例数据集")
                    data_type_param = request.data.get('data_type', 'microseismic')
                    
                    if data_type_param == 'microseismic':
                        # 导出示例微震数据 - 只导出数值列
                        simulated_data = list(MicroseismicData.objects.filter(is_simulated=True).values(
                            'energy', 'magnitude'
                        ))
                    else:
                        # 导出示例支架阻力数据 - 只导出数值列
                        simulated_data = list(SupportResistanceData.objects.filter(is_simulated=True).values(
                            'resistance_value'
                        ))
                    
                    if not simulated_data:
                        print("ERROR - 没有示例数据")
                        return Response({"error": "没有可用的示例数据。请先生成模拟数据。"}, status=status.HTTP_400_BAD_REQUEST)
                    
                    df = pd.DataFrame(simulated_data)
                    print(f"DEBUG - 示例数据导出前检查: {df.shape}, 列: {df.columns.tolist()}, 数据类型: {df.dtypes.to_dict()}")
                    
                    # 创建临时文件
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8')
                    df.to_csv(temp_file.name, index=False)
                    temp_file_path = temp_file.name
                    temp_file.close()
                    print(f"DEBUG - 示例数据已导出到: {temp_file_path}, 共 {len(df)} 条记录")
                    
                else:
                    # 常规数据集：通过 UploadedFile 获取关联数据
                    uploaded_file = UploadedFile.objects.get(id=dataset_id)
                    print(f"INFO - 找到上传文件: {uploaded_file.filename}, 类型: {uploaded_file.data_type}")
                    
                    # 根据数据类型获取关联数据 - 只导出数值列
                    if uploaded_file.data_type == 'microseismic':
                        dataset_data = list(MicroseismicData.objects.filter(uploaded_file=uploaded_file).values(
                            'energy', 'magnitude'
                        ))
                    else:  # support_resistance
                        dataset_data = list(SupportResistanceData.objects.filter(uploaded_file=uploaded_file).values(
                            'resistance_value'
                        ))
                    
                    if not dataset_data:
                        print(f"ERROR - 数据集 {dataset_id} 没有关联数据")
                        return Response({"error": "该数据集没有可用数据。"}, status=status.HTTP_400_BAD_REQUEST)
                    
                    df = pd.DataFrame(dataset_data)
                    print(f"DEBUG - 数据集导出前检查: {df.shape}, 列: {df.columns.tolist()}, 数据类型: {df.dtypes.to_dict()}")
                    
                    # 创建临时文件
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8')
                    df.to_csv(temp_file.name, index=False)
                    temp_file_path = temp_file.name
                    temp_file.close()
                    print(f"DEBUG - 数据已导出到: {temp_file_path}, 共 {len(df)} 条记录")
                    
            except UploadedFile.DoesNotExist:
                print(f"ERROR - 数据集不存在: {dataset_id}")
                return Response({"error": "指定的数据集不存在。"}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                print(f"ERROR - 获取数据集失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return Response({"error": f"获取数据集失败: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print(f"ERROR - 缺少数据源, file={file}, dataset_id={dataset_id}")
            return Response({"error": "缺少数据文件或数据集ID。"}, status=status.HTTP_400_BAD_REQUEST)
        
        task_id = start_training_session_async(config, temp_file_path)
        return Response({"task_id": str(task_id)}, status=status.HTTP_202_ACCEPTED)

class TrainingStatusView(APIView):
    # ... (此视图保持不变)
    def get(self, request, task_id, *args, **kwargs):
        run = get_object_or_404(TrainingRun, pk=task_id)
        response_data = { 'task_id': run.task_id, 'status': run.status, 'message': run.status_message, 'error': run.error_message }
        return Response(response_data, status=status.HTTP_200_OK)

class TrainingResultView(APIView):
    """
    获取训练结果，增加了详细的日志记录来排查文件读取问题。
    """
    def get(self, request, task_id, *args, **kwargs):
        print(f"\n--- [Task ID: {task_id}] 开始获取结果 ---")
        run = get_object_or_404(TrainingRun, pk=task_id, status=TrainingRun.Status.SUCCESS)
        results = ModelResult.objects.filter(training_run=run)
        
        response_data = []
        for result in results:
            plot_data = None
            print(f"正在处理模型: {result.model_name}")
            
            # 1. 检查数据库中是否有文件路径
            if result.plot_file_path:
                print(f"  数据库中找到文件路径: {result.plot_file_path}")
                
                # 2. 构建完整的文件绝对路径
                plot_full_path = os.path.join(settings.MEDIA_ROOT, result.plot_file_path)
                print(f"  尝试读取的绝对路径: {plot_full_path}")

                # 3. 检查文件是否存在
                if os.path.exists(plot_full_path):
                    print(f"  文件存在。正在尝试读取...")
                    try:
                        with open(plot_full_path, 'r') as f:
                            plot_data = json.load(f)
                        # 4. 检查读取到的数据是否有效
                        if plot_data and 'actuals' in plot_data and 'predictions' in plot_data:
                            print(f"  成功加载并解析JSON数据。")
                        else:
                            print(f"  [警告] JSON文件内容格式不正确，缺少'actuals'或'predictions'键。")
                            plot_data = None # 重置为None
                    except json.JSONDecodeError as e:
                        print(f"  [错误] JSON文件解析失败: {e}")
                        plot_data = None # 确保在出错时为None
                else:
                    print(f"  [错误] 文件在指定路径不存在！")
            else:
                print(f"  数据库中未找到文件路径。")

            response_data.append({
                'model_name': result.model_name,
                'metrics': result.metrics,
                'plot_data': plot_data # 返回绘图数据或None
            })
            
        print(f"--- [Task ID: {task_id}] 结果准备完毕，即将发送给前端 ---")
        return Response(response_data, status=status.HTTP_200_OK)
