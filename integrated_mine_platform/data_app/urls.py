from django.urls import path
from . import views

urlpatterns = [
    # 文件上传
    path('upload/', views.FileUploadView.as_view(), name='file_upload'),
    
    # 数据列表
    path('microseismic/', views.MicroseismicDataListView.as_view(), name='microseismic_list'),
    path('support-resistance/', views.SupportResistanceDataListView.as_view(), name='support_resistance_list'),
    
    # 上传文件列表
    path('uploaded-files/', views.UploadedFileListView.as_view(), name='uploaded_files_list'),
    
    # 统计信息
    path('statistics/', views.DataStatisticsView.as_view(), name='data_statistics'),
    
    # 数据集管理（用于分析模块）
    path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    path('datasets/data/', views.DatasetDataView.as_view(), name='dataset_data'),
    path('datasets/stations/', views.StationListView.as_view(), name='station_list'),
    
    # 演示数据管理
    path('demo-data/', views.DemoDataManagementView.as_view(), name='demo_data_management'),
]
