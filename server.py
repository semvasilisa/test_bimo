from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from tavily import TavilyClient
from datetime import datetime

# tavily key defined
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise RuntimeError("TAVILY_API_KEY is not set!")

client = TavilyClient(api_key=tavily_api_key)

# grow key defined
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set!")

# secure key 
bimo_token=os.getenv("BIMO_TOKEN")

if not bimo_token:
    raise RuntimeError("BIMO_TOKEN is not set!")

today = datetime.now().strftime("%Y-%m-%d") # for groq to know the current date

app = FastAPI() # creates the application that registers and handles your API routes

# describe what data our server expects to receive, inheriting from basemodel get our class data-checking features
class AskRequest(BaseModel):
    prompt: str

@app.post("/ask")
def ask_question(request: AskRequest): # here we mention class name so fastapi knows to expect a json body containing prompt
    question = request.prompt
    context = search_web(question)

    print("TAVILY CONTEXT:")
    print(context)

    answer = ask_groq(question, context)
    return {"answer": answer}

# method for a tavily search context
def search_web(user_query):
    response = client.search(
        query=user_query,
        search_depth="basic",
        max_results=3
    )

    snippets = []

    for result in response.get("results", []):
        snippet = result.get("content","") # response returns different fields and we extract only the content field, if it exists, otherwise we return an empty string
        
        if snippet:
            snippets.append(snippet)

    context = " ".join(snippets)
    return context

# method where we give the context and question to groq
def ask_groq(question, context):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model":"openai/gpt-oss-20b",
        "messages":[
            {
                "role": "system",
                "content": (
                    "You are BIMO, a helpful voice assistant. "
                    "Answer in 1-2 short, natural sentences suitable for speaking aloud. "
                    "Use only the information provided in the search results to answer factual questions. "
                    "Do not invent facts or make unsupported assumptions. "
                    "If the search results do not contain enough reliable information to answer the question, clearly say you don't know. "
                    "If the search results contain conflicting information, acknowledge the conflict instead of guessing. "
                    "Focus on answering the user's actual question directly."
                    f"Today's date is {today}. Answer the user's question..."
                )  
            },
            {
                "role": "user",
                "content": (
                    f"Search results:\n" 
                    f"{context}\n\n"
                    f"Question:\n" 
                    f"{question}"
                )
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)

    print("Status code:", response.status_code)
    response.raise_for_status()

    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    return answer