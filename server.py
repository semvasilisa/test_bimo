from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# describe what data our server expects to receive, inheriting from basemodel get our class data-checking features
class AskRequest(BaseModel):
    prompt: str

@app.post("/ask")
def ask_question(request: AskRequest): # here we mention class name so fastapi knows to expect a json body containing prompt
    return {"answer" : f"You asked: {request.prompt}"}
