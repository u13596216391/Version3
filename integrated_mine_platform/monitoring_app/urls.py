from django.urls import path
from .views import (
    RealtimeDataView,
    SimulatorView,
    generate_simulator_data_now,
    dashboard_data,
)

urlpatterns = [
    path('realtime/', RealtimeDataView.as_view(), name='realtime-data'),
    path('simulator/', SimulatorView.as_view(), name='simulator'),
    path('simulator/generate/', generate_simulator_data_now, name='simulator-generate'),
    path('dashboard/', dashboard_data, name='dashboard-data'),
]
