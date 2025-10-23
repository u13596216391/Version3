from django.urls import path
from .views import RealtimeDataView

urlpatterns = [
    path('realtime/', RealtimeDataView.as_view(), name='realtime-data'),
]
