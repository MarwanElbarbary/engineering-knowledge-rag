"""Connect retrieval -> prompt -> LLM into one RAG question-answering function."""
from src.retrivar.retriever import retrieve_documents
from src.generation.prompt import build_prompt
from src.generation.llm import get_llm
from src.config.setting import TOP_K


def ask_question(question, top_k=TOP_K, retrieval_method="similarity"):
    question = question.strip()
    if not question:
        raise ValueError("Please enter a non-empty question.")

    retrieved_chunks = retrieve_documents(
        query=question,
        k=top_k,
        method=retrieval_method,
    )
    if not retrieved_chunks:
        return "I don't know.", []

    prompt = build_prompt(question, retrieved_chunks)
    response = get_llm().invoke(prompt)
    return response.content, retrieved_chunks
