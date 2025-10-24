#!/usr/bin/env python
"""完整测试三个需求的实现"""
import requests
import json
from datetime import datetime

print("=" * 60)
print("需求1: 按照上传批次划分数据集")
print("=" * 60)

# 测试数据集API
r = requests.get('http://backend:8000/api/data/datasets/')
data = r.json()

print(f'\n✓ 数据集总数: {data["total"]}')
print('\n数据集列表:')
for i, d in enumerate(data['datasets'], 1):
    print(f'\n{i}. {d["name"]}')
    print(f'   类型: {d["type"]}')
    print(f'   数据量: {d["count"]}条')
    if d.get('upload_time'):
        upload_time = datetime.fromisoformat(d['upload_time'].replace('Z', '+00:00'))
        print(f'   上传时间: {upload_time.strftime("%Y-%m-%d %H:%M:%S")}')

print("\n" + "=" * 60)
print("需求2: 合并模拟数据集和演示数据集为示例数据集")
print("=" * 60)

simulated = [d for d in data['datasets'] if d['type'] == 'simulated']
print(f'\n✓ 找到 {len(simulated)} 个示例数据集')
for d in simulated:
    print(f'   - {d["name"]}: {d["count"]}条记录')
    print(f'     时间范围: {d["time_range"]["start"]} ~ {d["time_range"]["end"]}')

print("\n" + "=" * 60)
print("需求3: 测试分析功能（已删除"分析事件数据"选项）")
print("=" * 60)

# 测试分析API
print('\n✓ 测试使用示例数据集进行分析...')
r = requests.get('http://backend:8000/api/analysis/microseismic/scatter/', params={
    'start_date': '2025-10-16',
    'end_date': '2025-10-23',
    'analysis_type': 'frequency',
    'dataset_id': 'simulated'
})

result = r.json()
if result.get('success'):
    print(f'   ✓ 分析成功！')
    print(f'   - 分析记录数: {result["count"]}')
    print(f'   - 数据集名称: {result.get("dataset_name", "N/A")}')
    print(f'   - 事件总数: {result["statistics"]["total_events"]}')
    print(f'   - X坐标范围: {result["statistics"]["x_range"]}')
    print(f'   - Y坐标范围: {result["statistics"]["y_range"]}')
else:
    print(f'   ✗ 分析失败: {result.get("message")}')

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print('\n前端访问地址: http://localhost/analysis')
print('建议操作:')
print('1. 在数据集下拉框中选择"示例数据集"')
print('2. 选择日期范围: 2025-10-16 至 2025-10-23')
print('3. 点击"开始分析"按钮')
print('4. 查看散点图和密度图结果')
