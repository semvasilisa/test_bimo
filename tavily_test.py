import os
from tavily import TavilyClient

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise RuntimeError("TAVILY_API_KEY is not set!")

client = TavilyClient(api_key=api_key)

# response = client.search("What is the weather in Paris today?")

response = client.search(
    query="What is the weather in Paris today?",
    search_depth="basic",
    max_results=5
)

for result in response.get("results", []):
    print("Title:", result.get("title"))
    print("URL:", result.get("url"))
    print("Content:", result.get("content"))
    print()

