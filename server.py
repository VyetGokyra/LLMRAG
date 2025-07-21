from fastapi import FastAPI

from db import load_session_history, load_all_session_ids
from message_history_store import get_session_history
from rag_chain import conversational_rag_chain, invoke_and_save
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/chat")
async def chat(question: str, session_id: str):
    return invoke_and_save(session_id, question)


@app.get("/chat-history")
async def get_chat_history(session_id: str):
    return load_session_history(session_id)


@app.get("/all-session-ids")
async def get_all_session_ids():
    return load_all_session_ids()
