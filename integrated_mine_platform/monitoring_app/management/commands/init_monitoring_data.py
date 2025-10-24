"""
初始化监控数据 - 预填充和启动自动模拟器
"""
from django.core.management.base import BaseCommand
from monitoring_app.models import MonitoringData
from monitoring_app.simulator import get_simulator
from django.utils import timezone
from datetime import timedelta


class Command(BaseCommand):
    help = '初始化监控数据：预填充历史数据'

    def add_arguments(self, parser):
        parser.add_argument(
            '--initial-count',
            type=int,
            default=20,
            help='每种数据类型的初始数据点数量（默认20）'
        )

    def handle(self, *args, **options):
        initial_count = options['initial_count']
        
        # 检查是否已有数据
        existing_count = MonitoringData.objects.filter(is_simulated=True).count()
        
        if existing_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f'已存在 {existing_count} 条模拟数据，跳过初始化')
            )
            return
        
        self.stdout.write('开始生成初始监控数据...')
        
        simulator = get_simulator()
        data_list = []
        
        # 生成带时间戳的历史数据（过去1小时）
        now = timezone.now()
        time_interval = timedelta(minutes=3)  # 每3分钟一个数据点
        
        for i in range(initial_count):
            timestamp = now - time_interval * (initial_count - i)
            # 简单模拟：每次生成4种数据类型
            for data_type in ['microseismic', 'support_resistance', 'gas', 'temperature']:
                data = simulator.generate_single_data(data_type=data_type)
                data['timestamp'] = timestamp
                data_list.append(data)
        
        # 批量创建
        monitoring_objects = [MonitoringData(**data) for data in data_list]
        created = MonitoringData.objects.bulk_create(monitoring_objects)
        
        self.stdout.write(
            self.style.SUCCESS(f'✓ 成功生成 {len(created)} 条初始监控数据')
        )
