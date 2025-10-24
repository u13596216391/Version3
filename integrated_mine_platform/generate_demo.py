import requests
import json

# 生成演示数据
url = 'http://localhost:8000/api/data/demo-data/'
data = {
    'count': 200,
    'days': 14
}

response = requests.post(url, json=data)
print(f"状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
