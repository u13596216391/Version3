from django.urls import path
from .views import (
    DashboardOverviewView,
    StatisticsUpdateView,
    AlertListView,
    DataViewAPIView
)

urlpatterns = [
    path('overview/', DashboardOverviewView.as_view(), name='dashboard-overview'),
    path('statistics/update/', StatisticsUpdateView.as_view(), name='statistics-update'),
    path('alerts/', AlertListView.as_view(), name='alert-list'),
    path('data-view/', DataViewAPIView.as_view(), name='data-view'),
]
