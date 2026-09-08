# print("Hello")

import requests

url = "http://localhost:11434/api/chat"

data = {
    "model": "llama3.2",
    "messages": [
        {"role": "user", "content": "What is an API? Answer in one sentence."}
    ],
    "stream": False
}

response = requests.post(url, json=data)
result = response.json()

print("Status code:", response.status_code)
print("Model used: ", result["model"])
print("Reply:      ", result["message"]["content"])