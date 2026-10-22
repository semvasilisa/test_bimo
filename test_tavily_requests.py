import os
import requests

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise RuntimeError("TAVILY_API_KEY is not set!")

url = "https://api.tavily.com/search"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "query": "What is the weather in Paris today?",
    "search_depth": "basic",
    "max_results": 3
}

response = requests.post(url, headers=headers, json=data, timeout=30)

response.raise_for_status() #  raises an HTTPError when the response status code indicates an HTTP error

search_data = response.json()

for result in search_data.get("results", []):
    print("Content:", result.get("content"))
    print()