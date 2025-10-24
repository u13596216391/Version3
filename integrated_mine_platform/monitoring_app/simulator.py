"""
数据模拟器 - 生成各类监控数据
"""
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
from django.utils import timezone


class DataSimulator:
    """监控数据模拟器"""
    
    # 监测位置定义
    LOCATIONS = {
        'microseismic': ['1号工作面', '2号工作面', '3号工作面', '主巷道', '辅助巷道'],
        'support_resistance': ['支架A1', '支架A2', '支架B1', '支架B2', '支架C1'],
        'gas': ['监测点1', '监测点2', '监测点3', '回风巷', '工作面'],
        'temperature': ['工作面', '主巷道', '变电所', '泵房', '井底'],
        'humidity': ['工作面', '主巷道', '采煤面', '掘进面', '井底'],
    }
    
    # 数据范围定义 (最小值, 最大值, 异常阈值)
    DATA_RANGES = {
        'microseismic': {
            'min': 0.0,
            'max': 5.0,
            'abnormal_threshold': 3.5,
            'unit': '级',
        },
        'support_resistance': {
            'min': 10.0,
            'max': 50.0,
            'abnormal_threshold': 45.0,
            'unit': 'MPa',
        },
        'gas': {
            'min': 0.0,
            'max': 1.5,
            'abnormal_threshold': 1.0,
            'unit': '%',
        },
        'temperature': {
            'min': 15.0,
            'max': 35.0,
            'abnormal_threshold': 32.0,
            'unit': '°C',
        },
        'humidity': {
            'min': 40.0,
            'max': 95.0,
            'abnormal_threshold': 90.0,
            'unit': '%',
        },
    }
    
    def __init__(self):
        """初始化模拟器"""
        self.last_values = {}  # 存储上一次的值,用于生成连续变化的数据
    
    def generate_single_data(self, data_type: str, location: str = None) -> Dict[str, Any]:
        """
        生成单条模拟数据
        
        Args:
            data_type: 数据类型
            location: 监测位置(可选,不提供则随机选择)
        
        Returns:
            包含数据的字典
        """
        if data_type not in self.DATA_RANGES:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        # 获取数据范围配置
        config = self.DATA_RANGES[data_type]
        
        # 选择位置
        if location is None:
            location = random.choice(self.LOCATIONS[data_type])
        
        # 生成连续变化的值
        key = f"{data_type}_{location}"
        if key in self.last_values:
            # 基于上次的值进行小幅度变化
            last_value = self.last_values[key]
            change_range = (config['max'] - config['min']) * 0.1  # 最大变化10%
            change = random.uniform(-change_range, change_range)
            value = last_value + change
            # 确保在范围内
            value = max(config['min'], min(config['max'], value))
        else:
            # 首次生成,使用正态分布
            mean = (config['min'] + config['max']) / 2
            std = (config['max'] - config['min']) / 6
            value = random.gauss(mean, std)
            value = max(config['min'], min(config['max'], value))
        
        self.last_values[key] = value
        
        # 判断是否异常
        is_abnormal = value >= config['abnormal_threshold']
        
        # 偶尔生成异常值 (5%概率)
        if random.random() < 0.05:
            is_abnormal = True
            value = random.uniform(config['abnormal_threshold'], config['max'])
        
        return {
            'timestamp': timezone.now(),
            'data_type': data_type,
            'location': location,
            'value': round(value, 2),
            'unit': config['unit'],
            'is_abnormal': is_abnormal,
            'is_simulated': True,
        }
    
    def generate_batch_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        批量生成模拟数据
        
        Args:
            count: 每种数据类型生成的数量
        
        Returns:
            数据列表
        """
        all_data = []
        
        for data_type in self.DATA_RANGES.keys():
            locations = self.LOCATIONS[data_type]
            # 为每个位置生成数据
            for _ in range(min(count, len(locations))):
                location = random.choice(locations)
                data = self.generate_single_data(data_type, location)
                all_data.append(data)
        
        return all_data
    
    def generate_time_series_data(
        self,
        data_type: str,
        location: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """
        生成时间序列模拟数据
        
        Args:
            data_type: 数据类型
            location: 监测位置
            start_time: 开始时间
            end_time: 结束时间
            interval_minutes: 时间间隔(分钟)
        
        Returns:
            时间序列数据列表
        """
        data_list = []
        current_time = start_time
        
        while current_time <= end_time:
            data = self.generate_single_data(data_type, location)
            data['timestamp'] = current_time
            data_list.append(data)
            current_time += timedelta(minutes=interval_minutes)
        
        return data_list
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        生成大屏展示数据
        
        Returns:
            包含各类统计数据的字典
        """
        # 生成最近的监控数据
        recent_data = self.generate_batch_data(count=5)
        
        # 统计异常数据
        abnormal_count = sum(1 for d in recent_data if d['is_abnormal'])
        
        # 按类型统计
        type_stats = {}
        for data_type in self.DATA_RANGES.keys():
            type_data = [d for d in recent_data if d['data_type'] == data_type]
            if type_data:
                values = [d['value'] for d in type_data]
                type_stats[data_type] = {
                    'avg': round(sum(values) / len(values), 2),
                    'max': round(max(values), 2),
                    'min': round(min(values), 2),
                    'count': len(values),
                }
        
        return {
            'total_count': len(recent_data),
            'abnormal_count': abnormal_count,
            'normal_count': len(recent_data) - abnormal_count,
            'type_stats': type_stats,
            'recent_data': recent_data[:10],  # 最近10条
            'timestamp': timezone.now(),
        }
    
    def reset(self):
        """重置模拟器状态"""
        self.last_values.clear()


# 全局模拟器实例
_simulator = DataSimulator()


def get_simulator() -> DataSimulator:
    """获取模拟器单例"""
    return _simulator
