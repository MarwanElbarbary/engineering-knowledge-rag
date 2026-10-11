"""Build the prompt used in our working Kafka RAG example."""


def build_prompt(query, results):
    context = "\n\n".join(
        f"""Source: {doc.metadata.get('document_id')}
Version: {doc.metadata.get('version')}
Content:
{doc.page_content}
"""
        for doc in results
    )

    return f"""You are an engineering knowledge assistant.

Instructions:
1. Answer using only the provided context.
2. Prefer current accepted decisions over superseded decisions.
3. If the context does not contain the answer, say you don't know.
4. Cite the document IDs used in your answer.

Context:
{context}

Question:
{query}

Answer:
"""
