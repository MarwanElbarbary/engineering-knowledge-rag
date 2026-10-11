"""Connect to the same Groq model; API key is read from a local .env file."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.config.setting import BASE_DIR, LLM_MODEL


@lru_cache(maxsize=1)
def get_llm():
    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and set your own key."
        )

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        reasoning_format="hidden",
    )
