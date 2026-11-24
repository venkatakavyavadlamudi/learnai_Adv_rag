
import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LearnAI Role-Based RAG API")


# CORS configuration
frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Allowed application roles
ALLOWED_ROLES = {
    "Student",
    "Mentor",
    "Trainer",
    "Admin",
    "Super Admin"
}


# Request model
class ChatRequest(BaseModel):
    question: str
    role: str
    chat_history: list = Field(default_factory=list)


# Convert frontend chat history into LangChain messages
def convert_chat_history(chat_history):

    history = []

    for message in chat_history:

        if isinstance(message, str):
            history.append(
                HumanMessage(content=message)
            )

        elif isinstance(message, dict):

            role = message.get("role", "user")
            content = message.get("content", "")

            if not isinstance(content, str) or not content.strip():
                continue

            if role in ["user", "human"]:
                history.append(
                    HumanMessage(content=content)
                )

            elif role in ["assistant", "ai"]:
                history.append(
                    AIMessage(content=content)
                )

    return history


# Health check
@app.get("/")
def home():
    return {
        "message": "LearnAI Role-Based RAG API is running"
    }


# Role-based chatbot endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    try:

        # Validate question
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Validate role
        if request.role not in ALLOWED_ROLES:
            raise HTTPException(
                status_code=400,
                detail="Invalid role. Allowed roles: Student, Mentor, Trainer, Admin, Super Admin"
            )

        # Convert chat history
        history = convert_chat_history(
            request.chat_history
        )

        logger.info("Received role-based chatbot request")
        logger.info("Selected role: %s", request.role)
        logger.info("Chat history messages: %s", len(history))

        # Invoke role-based RAG pipeline
        response = rag_chain.invoke({
            "input": request.question,
            "chat_history": history,
            "role": request.role
        })

        # Extract answer
        if isinstance(response, dict):
            answer = response.get(
                "answer",
                response.get("output", str(response))
            )
        else:
            answer = str(response)

        return {
            "answer": answer,
            "role": request.role
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Role-based chatbot processing failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question"
        )
