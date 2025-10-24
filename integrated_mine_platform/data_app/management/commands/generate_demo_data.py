"""
生成演示数据的管理命令
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from data_app.models import MicroseismicData
from datetime import datetime, timedelta
import random
import numpy as np


class Command(BaseCommand):
    help = '生成微震演示数据'

    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=500,
            help='生成数据条数 (默认500)'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='时间跨度天数 (默认30天)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='清除现有演示数据'
        )

    def handle(self, *args, **options):
        count = options['count']
        days = options['days']
        clear = options['clear']

        # 清除现有演示数据
        if clear:
            deleted_count = MicroseismicData.objects.filter(is_simulated=True).count()
            MicroseismicData.objects.filter(is_simulated=True).delete()
            self.stdout.write(self.style.SUCCESS(f'已清除 {deleted_count} 条演示数据'))
            if not count:
                return

        # 生成演示数据
        self.stdout.write(f'开始生成 {count} 条演示数据...')
        
        # 时间范围
        end_time = timezone.now()
        start_time = end_time - timedelta(days=days)
        
        # 空间范围 (模拟一个矿区)
        x_range = (0, 1700)  # 工作面长度
        y_range = (-200, 200)  # 巷道宽度
        z_range = (-800, -300)  # 深度范围
        
        # 生成数据集中的"热点区域" (模拟应力集中区)
        hotspots = [
            {'center': (850, 0, -500), 'radius': 200, 'weight': 0.4},
            {'center': (300, 50, -400), 'radius': 150, 'weight': 0.3},
            {'center': (1400, -30, -600), 'radius': 180, 'weight': 0.3},
        ]
        
        records = []
        for i in range(count):
            # 生成时间戳
            time_delta = timedelta(seconds=random.randint(0, int(days * 24 * 3600)))
            timestamp = start_time + time_delta
            
            # 决定是否在热点区域 (70%概率)
            if random.random() < 0.7:
                hotspot = random.choices(hotspots, weights=[h['weight'] for h in hotspots])[0]
                # 在热点周围生成，使用正态分布
                event_x = np.random.normal(hotspot['center'][0], hotspot['radius'] / 3)
                event_y = np.random.normal(hotspot['center'][1], hotspot['radius'] / 3)
                event_z = np.random.normal(hotspot['center'][2], hotspot['radius'] / 3)
                # 热点区域能量更高
                energy_base = random.uniform(1e6, 1e8)
            else:
                # 随机分布
                event_x = random.uniform(*x_range)
                event_y = random.uniform(*y_range)
                event_z = random.uniform(*z_range)
                energy_base = random.uniform(1e4, 1e6)
            
            # 限制在范围内
            event_x = max(x_range[0], min(x_range[1], event_x))
            event_y = max(y_range[0], min(y_range[1], event_y))
            event_z = max(z_range[0], min(z_range[1], event_z))
            
            # 计算能量和震级 (使用对数关系)
            energy = energy_base * random.uniform(0.5, 2.0)
            magnitude = (np.log10(energy) - 4.8) / 1.5  # 里氏震级公式简化版
            
            # 生成事件ID
            event_id = f'DEMO_{timestamp.strftime("%Y%m%d")}_{i:04d}'
            
            record = MicroseismicData(
                timestamp=timestamp,
                event_id=event_id,
                event_x=round(event_x, 2),
                event_y=round(event_y, 2),
                event_z=round(event_z, 2),
                energy=energy,
                magnitude=round(magnitude, 2),
                locate_mw=round(magnitude + random.uniform(-0.2, 0.2), 2),
                locate_err=round(random.uniform(1, 10), 2),
                velocity=round(random.uniform(3000, 6000), 1),
                source_file='演示数据',
                is_simulated=True
            )
            records.append(record)
            
            # 每100条显示进度
            if (i + 1) % 100 == 0:
                self.stdout.write(f'已生成 {i + 1}/{count} 条...')
        
        # 批量创建
        MicroseismicData.objects.bulk_create(records, batch_size=500)
        
        self.stdout.write(self.style.SUCCESS(
            f'\n成功生成 {count} 条演示数据！\n'
            f'时间范围: {start_time.strftime("%Y-%m-%d")} ~ {end_time.strftime("%Y-%m-%d")}\n'
            f'空间范围: X({x_range[0]}-{x_range[1]}) Y({y_range[0]}-{y_range[1]}) Z({z_range[0]}-{z_range[1]})'
        ))
