from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from handlers.base import BaseHandler
from typing import Optional

router = APIRouter()

class ChatModel(BaseModel):
    query: str
    model: str = "gpt-4o-mini"
    temperature: float
    chat_history: list[str] = [] # Example input: [("You are a helpful assistant.", "What is your name?")]

@router.post("/chat")
async def chat( 
    chat_model: ChatModel,
):
    available_models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]

    if chat_model.model not in available_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Please select a valid model from the list of available models: \n{str(available_models)}")
    
    if chat_model.temperature < 0.0 or chat_model.temperature > 2.0:
        raise HTTPException(status_code=400, detail="Invalid temperature value. Please select a value between 0.0 and 2.0")

    handler = BaseHandler(chat_model=chat_model.model, openai_chat_temperature=chat_model.temperature)
    response = handler.chat(
        "",
        chat_model.query, 
        chat_model.chat_history
    )
    return {"response": response}
    