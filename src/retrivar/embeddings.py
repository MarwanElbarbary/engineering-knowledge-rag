"""Same Hugging Face embedding model used in our experiment (384 dimensions)."""
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.setting import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
