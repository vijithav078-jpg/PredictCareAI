import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "temperature": 60,
    "process_temperature": 65,
    "rpm": 900,
    "torque": 25,
    "tool_wear": 30
}

response = requests.post(url, json=data)

print(response.json())