from django.urls import path
from .views import (
    MicroseismicScatterView,
    MicroseismicDensityView,
    SupportDWTAnalysisView,
    SupportWaveletComparisonView,
    get_available_stations,
    get_analysis_history,
)

app_name = 'analysis_app'

urlpatterns = [
    # 微震分析
    path('microseismic/scatter/', MicroseismicScatterView.as_view(), name='microseismic-scatter'),
    path('microseismic/density/', MicroseismicDensityView.as_view(), name='microseismic-density'),
    
    # 支架阻力分析
    path('support/dwt/', SupportDWTAnalysisView.as_view(), name='support-dwt'),
    path('support/wavelet-comparison/', SupportWaveletComparisonView.as_view(), name='support-wavelet-comparison'),
    
    # 工具接口
    path('stations/', get_available_stations, name='available-stations'),
    path('history/', get_analysis_history, name='analysis-history'),
]
