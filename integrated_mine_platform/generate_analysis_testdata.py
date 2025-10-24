"""
生成分析应用的测试数据
包括微震事件和支架阻力数据
"""
import os
import django
import sys
from datetime import datetime, timedelta
import random
import numpy as np

# 设置Django环境
sys.path.insert(0, '/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'integrated_mine.settings')
django.setup()

from analysis_app.models import MicroseismicEvent, SupportResistance


def generate_microseismic_events(days=30, events_per_day=50):
    """生成微震事件测试数据"""
    print(f"生成微震事件数据: {days}天, 每天{events_per_day}个事件...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    events = []
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        
        for j in range(events_per_day):
            # 生成随机坐标(模拟工作面区域)
            x_coord = random.uniform(0, 1700)  # 0-1700m
            y_coord = random.uniform(-50, 150)  # -50-150m
            
            # 生成随机能量和震级
            magnitude = random.uniform(1.0, 4.5)
            energy = 10 ** (1.5 * magnitude + 4.8)  # 简化的能量计算
            
            # 添加时间偏移
            timestamp = current_date + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            # 随机选择数据类型
            data_type = random.choice(['frequency', 'frequency', 'energy'])  # frequency更常见
            
            event = MicroseismicEvent(
                timestamp=timestamp,
                x_coord=x_coord,
                y_coord=y_coord,
                magnitude=magnitude,
                energy=energy,
                data_type=data_type,
                is_simulated=True
            )
            events.append(event)
    
    # 批量创建
    MicroseismicEvent.objects.bulk_create(events, batch_size=500)
    print(f"✓ 已创建 {len(events)} 个微震事件")


def generate_support_resistance_data(days=30, stations=5, samples_per_day=288):
    """生成支架阻力测试数据"""
    print(f"生成支架阻力数据: {days}天, {stations}个测站, 每天{samples_per_day}个样本...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    resistance_records = []
    
    for station_idx in range(1, stations + 1):
        station_id = f"STATION_{station_idx:03d}"
        
        # 每个测站的基础阻力值不同
        base_resistance = random.uniform(15.0, 25.0)
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            
            # 每天采样samples_per_day次 (每5分钟一次: 288 = 24*60/5)
            for j in range(samples_per_day):
                # 添加周期性变化和随机噪声
                time_factor = j / samples_per_day * 2 * np.pi
                periodic_component = 3 * np.sin(time_factor) + 1.5 * np.sin(3 * time_factor)
                noise = np.random.normal(0, 0.5)
                
                # 偶尔添加异常高压事件
                if random.random() < 0.02:  # 2%概率
                    spike = random.uniform(8, 15)
                else:
                    spike = 0
                
                resistance_value = base_resistance + periodic_component + noise + spike
                
                # 确保阻力值为正
                resistance_value = max(resistance_value, 5.0)
                
                timestamp = current_date + timedelta(minutes=j * 5)
                
                record = SupportResistance(
                    timestamp=timestamp,
                    station_id=station_id,
                    resistance_value=resistance_value,
                    is_simulated=True
                )
                resistance_records.append(record)
    
    # 批量创建
    SupportResistance.objects.bulk_create(resistance_records, batch_size=1000)
    print(f"✓ 已创建 {len(resistance_records)} 条支架阻力记录")


def main():
    print("=" * 60)
    print("开始生成分析应用测试数据")
    print("=" * 60)
    
    # 清理现有数据
    print("\n清理现有测试数据...")
    MicroseismicEvent.objects.filter(is_simulated=True).delete()
    SupportResistance.objects.filter(is_simulated=True).delete()
    print("✓ 清理完成")
    
    print("\n" + "=" * 60)
    
    # 生成微震数据
    generate_microseismic_events(days=30, events_per_day=50)
    
    print("\n" + "=" * 60)
    
    # 生成支架阻力数据
    generate_support_resistance_data(days=30, stations=5, samples_per_day=288)
    
    print("\n" + "=" * 60)
    print("测试数据生成完成！")
    print("=" * 60)
    
    # 统计信息
    print("\n数据统计:")
    print(f"  - 微震事件总数: {MicroseismicEvent.objects.count()}")
    print(f"  - 支架阻力记录总数: {SupportResistance.objects.count()}")
    print(f"  - 测站数量: {SupportResistance.objects.values('station_id').distinct().count()}")


if __name__ == '__main__':
    main()
