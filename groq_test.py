import os
import requests

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set!")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model":"openai/gpt-oss-20b",
    "messages":[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer briefly."
        },
        {
            "role": "user",
            "content": "What is the weather in Paris today?"
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=30)

print("Status code:", response.status_code)
response.raise_for_status()

result = response.json()
answer = result["choices"][0]["message"]["content"]
print("Answer:", answer)