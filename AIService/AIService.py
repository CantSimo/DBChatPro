from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"detail": "Langchain Chatbot is Running!"}

from endpoints import (
    chat,
)

for endpoint in [chat]:
    app.include_router(endpoint.router)

if __name__ == "__main__":
    uvicorn.run('AIService:app', port=9092, reload=True)