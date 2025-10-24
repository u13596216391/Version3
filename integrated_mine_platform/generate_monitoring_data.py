"""
生成模拟监控数据的脚本
"""
import os
import sys
import django
from datetime import datetime, timedelta
import random

# 设置Django环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'integrated_mine.settings')
django.setup()

from monitoring_app.models import MonitoringData

def generate_data():
    """生成模拟监控数据"""
    
    # 清空现有的模拟数据
    MonitoringData.objects.filter(is_simulated=True).delete()
    
    now = datetime.now()
    data_to_create = []
    
    # 生成过去24小时的数据，每5分钟一条
    for i in range(288):  # 24 * 60 / 5 = 288
        timestamp = now - timedelta(minutes=i*5)
        
        # 微震数据
        data_to_create.append(MonitoringData(
            timestamp=timestamp,
            data_type='microseismic',
            location='3号工作面',
            value=round(random.uniform(1.5, 3.5), 2),
            unit='级',
            is_abnormal=random.random() < 0.1,  # 10%概率异常
            is_simulated=True
        ))
        
        # 支架阻力
        data_to_create.append(MonitoringData(
            timestamp=timestamp,
            data_type='support_resistance',
            location='ZJ-001',
            value=round(random.uniform(20, 30), 1),
            unit='MPa',
            is_abnormal=random.random() < 0.05,
            is_simulated=True
        ))
        
        # 瓦斯浓度
        data_to_create.append(MonitoringData(
            timestamp=timestamp,
            data_type='gas',
            location='回风巷道',
            value=round(random.uniform(0.3, 0.8), 2),
            unit='%',
            is_abnormal=random.random() < 0.08,
            is_simulated=True
        ))
        
        # 温度
        data_to_create.append(MonitoringData(
            timestamp=timestamp,
            data_type='temperature',
            location='主井',
            value=round(random.uniform(25, 32), 1),
            unit='℃',
            is_abnormal=random.random() < 0.03,
            is_simulated=True
        ))
    
    # 批量创建
    MonitoringData.objects.bulk_create(data_to_create)
    
    print(f"✅ 成功生成 {len(data_to_create)} 条模拟监控数据")
    
    # 显示统计信息
    for data_type, label in MonitoringData.DATA_TYPES:
        count = MonitoringData.objects.filter(data_type=data_type, is_simulated=True).count()
        print(f"   - {label}: {count} 条")

if __name__ == '__main__':
    generate_data()
