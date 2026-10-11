"""Project paths and the same model/settings we used in our first RAG."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
METADATA_FILE = BASE_DIR / "data" / "metadata" / "documents.json"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "engineering_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
