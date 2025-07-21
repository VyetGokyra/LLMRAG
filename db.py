from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from sqlmodel import Field, Session, SQLModel, create_engine, Text, Column, Relationship, select


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    role: str = Field(nullable=False)
    content: str = Field(nullable=False)
    session_id: str = Field(nullable=False)


engine = create_engine("sqlite:///sqlite.db")

SQLModel.metadata.create_all(engine)


# Function to save a single message
def save_message(session_id: str, role: str, content: str):
    with Session(engine) as session:
        session.add(ChatMessage(session_id=session_id, role=role, content=content))
        session.commit()


# Function to load chat history
def load_session_history(session_id: str) -> BaseChatMessageHistory:
    chat_history = ChatMessageHistory()
    with Session(engine) as session:
        statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
        results = session.exec(statement)
        for chat_message in results:
            chat_history.add_message({"role": chat_message.role, "content": chat_message.content})
        return chat_history


def load_all_session_ids() -> list[str]:
    with Session(engine) as session:
        statement = select(ChatMessage.session_id).distinct()
        return session.exec(statement).all()


