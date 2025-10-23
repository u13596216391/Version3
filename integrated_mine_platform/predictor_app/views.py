# predictor_app/views.py

import os
import tempfile
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers
from django.shortcuts import get_object_or_404
from django.conf import settings

from .models import TrainingRun, ModelResult
from .services import start_training_session_async

class StartTrainingView(APIView):
    # ... (此视图保持不变)
    parser_classes = [parsers.MultiPartParser]
    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        config_str = request.data.get('config')
        if not file or not config_str:
            return Response({"error": "缺少数据文件或config配置。"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            return Response({"error": "Config必须是有效的JSON字符串。"}, status=status.HTTP_400_BAD_REQUEST)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            for chunk in file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name
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
