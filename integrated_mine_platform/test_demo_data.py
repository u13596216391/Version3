#!/usr/bin/env python
"""测试演示数据生成"""
import os
import django
import random
import numpy as np
from datetime import timedelta

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'integrated_mine.settings')
django.setup()

from django.utils import timezone
from data_app.models import MicroseismicData

def generate_demo_data(count=100, days=7):
    """生成演示数据"""
    print(f"开始生成 {count} 条演示数据，时间跨度 {days} 天...")
    
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
    
    # 批量创建
    MicroseismicData.objects.bulk_create(records, batch_size=500)
    print(f"✓ 成功生成 {len(records)} 条演示数据")
    print(f"  时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    # 统计信息
    stats = MicroseismicData.objects.filter(is_simulated=True).aggregate(
        total=Count('id'),
        min_time=Min('timestamp'),
        max_time=Max('timestamp')
    )
    print(f"\n当前演示数据统计:")
    print(f"  总数量: {stats['total']}")
    print(f"  时间范围: {stats['min_time']} ~ {stats['max_time']}")

if __name__ == '__main__':
    from django.db.models import Count, Min, Max
    generate_demo_data(count=100, days=7)
