
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain


load_dotenv()

app = FastAPI(title="LearnAI Role-Based RAG API")

# CORS
origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([supabase_url, supabase_anon_key, service_role_key]):
    raise RuntimeError("Missing Supabase environment variables")

auth_client = create_client(
    supabase_url,
    supabase_anon_key
)

admin_client = create_client(
    supabase_url,
    service_role_key
)

security = HTTPBearer()

UNKNOWN_ANSWER = "I don't know based on the provided document."


# Request format
class ChatRequest(BaseModel):
    question: str
    chat_history: list = Field(default_factory=list)


# Verify logged-in user
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        response = auth_client.auth.get_user(token)

        if not response.user:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

        return response.user

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )


# Get role from learnai_users table
def get_user_role(user):
    try:
        response = (
            admin_client.table("learnai_users")
            .select("role")
            .eq("user_id", str(user.id))
            .single()
            .execute()
        )

        role = response.data["role"].strip().lower()

        role_mapping = {
            "student": "Student",
            "trainer": "Trainer",
            "mentor": "Mentor",
            "admin": "Admin",
            "super_admin": "Admin"
        }

        if role not in role_mapping:
            raise HTTPException(
                status_code=403,
                detail="Unauthorized role"
            )

        return role_mapping[role]

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=403,
            detail="User role not found"
        )


# Convert frontend history into LangChain messages
def convert_history(history):
    messages = []

    for item in history[-10:]:
        if item.get("role") == "user":
            messages.append(
                HumanMessage(content=item.get("content", ""))
            )

        elif item.get("role") == "assistant":
            messages.append(
                AIMessage(content=item.get("content", ""))
            )

    return messages


@app.get("/")
def home():
    return {
        "message": "LearnAI Role-Based RAG API is running"
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    user=Depends(get_current_user)
):
    role = get_user_role(user)

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    history = convert_history(request.chat_history)

    try:
        answer = rag_chain.invoke({
            "input": request.question,
            "chat_history": history,
            "role": role
        })

        return {
            "answer": answer,
            "role": role
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to process your question"
        )
