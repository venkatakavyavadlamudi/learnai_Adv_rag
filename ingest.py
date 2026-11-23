
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import vectorstore


DATA_DIR = Path("data")

ROLE_FILES = {
    "Student": "LearnAI_Student_Policy.pdf",
    "Trainer": "LearnAI_Trainer_Policy.pdf",
    "Mentor": "LearnAI_Mentor_Policy.pdf",
    "Admin": "LearnAI_Admin_Policy.pdf"
}

all_documents = []

# 1. Load PDF documents
for role, filename in ROLE_FILES.items():

    pdf_path = DATA_DIR / filename

    if not pdf_path.exists():
        print(f"Missing file: {filename}")
        continue

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    # Add role and source metadata
    for doc in documents:

        doc.metadata["role"] = role
        doc.metadata["source_file"] = filename

    all_documents.extend(documents)


if not all_documents:
    raise FileNotFoundError(
        "No PDF documents found in data folder."
    )


# 2. Split PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(all_documents)

print("Total chunks:", len(chunks))


# 3. Store chunks in Supabase
vectorstore.add_documents(chunks)

print("Successfully stored documents in Supabase!")
print("Total chunks uploaded:", len(chunks))
