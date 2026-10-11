"""Connect retrieval -> prompt -> LLM into one RAG question-answering function."""
from src.retrivar.retriever import retrieve_documents
from src.generation.prompt import build_prompt
from src.generation.llm import get_llm


def ask_question(question):
    question = question.strip()
    if not question:
        raise ValueError("Please enter a non-empty question.")

    retrieved_chunks = retrieve_documents(question)
    if not retrieved_chunks:
        return "I don't know.", []

    prompt = build_prompt(question, retrieved_chunks)
    response = get_llm().invoke(prompt)
    return response.content, retrieved_chunks
