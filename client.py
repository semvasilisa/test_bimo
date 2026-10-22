import requests

# sends http posr request to a server
response = requests.post(
    "http://localhost:8000/ask",
    json={"prompt": "Hello BIMO"}
)

data = response.json()
print(data["answer"])