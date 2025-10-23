from django.urls import path
from . import views

# 这个文件里的所有URL，都会自动带上 /api/predictor/ 的前缀
urlpatterns = [
    path('start-training/', views.StartTrainingView.as_view(), name='start-training'),
    path('status/<uuid:task_id>/', views.TrainingStatusView.as_view(), name='training-status'),
    path('results/<uuid:task_id>/', views.TrainingResultView.as_view(), name='training-result'),
]
