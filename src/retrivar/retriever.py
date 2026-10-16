"""Semantic similarity retrieval from the existing Chroma database."""
from src.config.setting import CHROMA_DIR, TOP_K
from src.retrivar.vector_store import get_vector_store


def retrieve_documents(query, k=TOP_K, method="similarity"):
    # Explicit error instead of querying an empty, newly created database.
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "Chroma database not built yet. Run: python -m src.retrivar.vector_store"
        )
    vector_store = get_vector_store()
    if method == "similarity":
        return vector_store.similarity_search(query=query, k=k)
    if method == "mmr":
        return vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=max(10, k * 3),
            lambda_mult=0.5,
        )
    raise ValueError(
        "Unsupported retrieval method. Use 'similarity' or 'mmr'."
    )
