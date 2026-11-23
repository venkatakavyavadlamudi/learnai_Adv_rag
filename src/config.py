
import os

from dotenv import load_dotenv
from supabase import create_client

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()

# 1. Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing in .env")

# 3. LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)

# 4. Supabase credentials
supabase_url = os.getenv("SUPABASE_URL")
service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not service_role_key:
    raise ValueError("Supabase credentials are missing")

# 5. Create Supabase client
supabase_client = create_client(
    supabase_url,
    service_role_key
)

# 6. Connect to Supabase vector store
vectorstore = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="data_storage",
    query_name="match_documents"
)

# 7. Role-based retriever
def get_retriever(role):

    allowed_roles = [
        "Student",
        "Trainer",
        "Mentor",
        "Admin"
    ]

    if role not in allowed_roles:
        raise ValueError("Invalid role")

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"role": role}
        }
    )

    return retriever
