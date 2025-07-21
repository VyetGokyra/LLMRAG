from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from db import load_session_history, save_message
import atexit

store = {}


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]
#
#
# Modify the get_session_history function to use the database
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]


def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])


atexit.register(save_all_sessions)
