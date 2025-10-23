import os
import json
import zipfile
import tempfile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers
from django.shortcuts import get_object_or_404
from django.conf import settings

from .models import MSTrainingRun, MSModelResult
from .services import start_ms_training_async

class StartMSTrainingView(APIView):
    # ... (此视图保持不变)
    parser_classes = [parsers.MultiPartParser]
    def post(self, request, *args, **kwargs):
        zip_file = request.FILES.get('ms_folder_zip')
        config_str = request.data.get('config')
        if not zip_file or not config_str:
            return Response({"error": "缺少微震数据压缩包或config配置。"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            return Response({"error": "Config必须是有效的JSON字符串。"}, status=status.HTTP_400_BAD_REQUEST)
        temp_dir = tempfile.mkdtemp()
        file_paths = {}
        ms_folder_path = os.path.join(temp_dir, 'ms_data')
        os.makedirs(ms_folder_path, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(ms_folder_path)
        file_paths['ms_folder'] = ms_folder_path
        task_id = start_ms_training_async(config, file_paths)
        return Response({"task_id": str(task_id)}, status=status.HTTP_202_ACCEPTED)

class MSTrainingStatusView(APIView):
    # ... (此视图保持不变)
    def get(self, request, task_id, *args, **kwargs):
        run = get_object_or_404(MSTrainingRun, pk=task_id)
        response_data = { 'task_id': run.task_id, 'status': run.status, 'message': run.status_message, 'error': run.error_message }
        return Response(response_data, status=status.HTTP_200_OK)

class MSTrainingResultView(APIView):
    """
    获取训练结果，现在会包含用于绘图的原始数据。
    """
    def get(self, request, task_id, *args, **kwargs):
        run = get_object_or_404(MSTrainingRun, pk=task_id, status=MSTrainingRun.Status.SUCCESS)
        results = MSModelResult.objects.filter(training_run=run)
        
        response_data = []
        for result in results:
            plot_data = None
            if result.plot_file_path:
                plot_full_path = os.path.join(settings.MEDIA_ROOT, result.plot_file_path)
                try:
                    with open(plot_full_path, 'r') as f:
                        plot_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"无法加载微震绘图数据: {e}")

            response_data.append({
                'model_name': result.model_name,
                'metrics': result.metrics,
                'plot_data': plot_data
            })
        return Response(response_data, status=status.HTTP_200_OK)