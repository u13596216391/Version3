from django.urls import path
from . import views

# 这个文件里的所有URL，都会自动带上 /api/microseismic_predictor/ 的前缀
urlpatterns = [
    path('start-training/', views.StartMSTrainingView.as_view(), name='ms-start-training'),
    path('status/<uuid:task_id>/', views.MSTrainingStatusView.as_view(), name='ms-training-status'),
    path('results/<uuid:task_id>/', views.MSTrainingResultView.as_view(), name='ms-training-result'),
]
