"""
集成矿山智能预测平台 - 主路由配置
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # API路由
    path('api/predictor/', include('predictor_app.urls')),  # 支架阻力预测
    path('api/microseismic/', include('microseismic_app.urls')),  # 微震预测
    path('api/monitoring/', include('monitoring_app.urls')),  # 实时监控
    path('api/dashboard/', include('dashboard_app.urls')),  # 数据大屏
]

# 开发环境下提供媒体文件服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
