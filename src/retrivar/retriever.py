"""Semantic similarity retrieval from the existing Chroma database."""
from src.config.setting import CHROMA_DIR, TOP_K
from src.retrivar.vector_store import get_vector_store


def retrieve_documents(query, k=TOP_K):
    # Explicit error instead of querying an empty, newly created database.
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "Chroma database not built yet. Run: python -m src.retrivar.vector_store"
        )
    vector_store = get_vector_store()
    return vector_store.similarity_search(query=query, k=k)
