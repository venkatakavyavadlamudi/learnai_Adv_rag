
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain

load_dotenv()

app = FastAPI(title="LearnAI RAG API")

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


# Request model
class ChatRequest(BaseModel):
    question: str
    chat_history: list = Field(default_factory=list)


# Health check
@app.get("/")
def home():
    return {
        "message": "LearnAI RAG API is running"
    }


# Chat endpoint without authentication
@app.post("/chat")
def chat(request: ChatRequest):

    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Convert chat history into LangChain messages
        history = []

        for message in request.chat_history:
            if message.get("role") == "user":
                history.append(
                    HumanMessage(content=message.get("content", ""))
                )

            elif message.get("role") == "assistant":
                history.append(
                    AIMessage(content=message.get("content", ""))
                )

        # Invoke RAG pipeline
        response = rag_chain.invoke({
            "input": request.question,
            "chat_history": history
        })

        # Extract answer
        if isinstance(response, dict):
            answer = response.get("answer", response.get("output", str(response)))
        else:
            answer = str(response)

        return {
            "answer": answer
        }

    except HTTPException:
        raise

    except Exception as e:
        print("Chatbot error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question"
        )
