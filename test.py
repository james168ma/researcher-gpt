import requests

data = {"query": "How to study a language effectively in 2023?"}
url = "http://127.0.0.1:8000"
print(requests.post(url, json=data).json())
