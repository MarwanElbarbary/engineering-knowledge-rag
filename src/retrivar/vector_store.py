"""Build and open the local Chroma database from the original 24 documents."""
from functools import lru_cache

from langchain_chroma import Chroma
from src.config.setting import CHROMA_DIR, COLLECTION_NAME
from src.ingestion.loader import load_documents
from src.ingestion.chunker import split_documents
from src.retrivar.embeddings import get_embedding_model


@lru_cache(maxsize=1)
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIR),
    )


def build_vector_database():
    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = get_vector_store()

    # Keep the same ID convention used in our working experiment.
    counts = {}
    chunk_ids = []
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        number = counts.get(doc_id, 0)
        chunk_ids.append(f"{doc_id}-chunk-{number}")
        counts[doc_id] = number + 1

    vector_store.add_documents(documents=chunks, ids=chunk_ids)
    stored_count = len(vector_store.get()["ids"])
    return len(documents), len(chunks), stored_count


if __name__ == "__main__":
    document_count, chunk_count, stored_count = build_vector_database()
    print("Files loaded (LangChain Documents):", document_count)
    print("Total Chunks:", chunk_count)
    print("Vector Database Created!")
    print("Total Stored Chunks:", stored_count)
