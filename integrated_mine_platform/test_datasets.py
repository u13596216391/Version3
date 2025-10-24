#!/usr/bin/env python
"""测试数据集API"""
import requests
import json

r = requests.get('http://backend:8000/api/data/datasets/')
data = r.json()

print(f'Total datasets: {data["total"]}')
print('\nFirst 5 datasets:')
for d in data['datasets'][:5]:
    print(f'  - {d["name"]} ({d["type"]}, {d["count"]}条)')
