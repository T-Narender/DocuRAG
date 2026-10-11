import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


FALLBACK_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "YOUR_DEFAULT_FALLBACK_API_KEY",
)

VECTORSTORE_DIR = BASE_DIR / "chroma_db_upload"

SUPPORTED_TYPES = [
    "pdf",
    "txt",
    "docx",
    "doc",
    "pptx",
    "ppt",
]

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

GEMINI_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

RETRIEVAL_K = 10
RERANK_TOP_N = 3