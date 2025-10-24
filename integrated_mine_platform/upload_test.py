#!/usr/bin/env python
"""测试文件上传"""
import requests

file_path = r'c:\Users\m1359\Desktop\tzb\Version3\old_version\test_data\csv\2024-09-28.csv'
url = 'http://localhost:8000/api/data/upload/'

with open(file_path, 'rb') as f:
    files = {'file': f}
    data = {'dataset_name': 'test_dataset_20240928'}
    response = requests.post(url, files=files, data=data)
    
print(f"状态码: {response.status_code}")
print(f"响应: {response.text}")
