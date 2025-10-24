import requests

# 测试文件上传
files = {
    'file': open('test_microseismic.csv', 'rb')
}
data = {
    'data_type': 'microseismic',
    'file_type': 'csv'
}

response = requests.post('http://localhost/api/data/upload/', files=files, data=data)
print(f"状态码: {response.status_code}")
print(f"响应: {response.text}")
