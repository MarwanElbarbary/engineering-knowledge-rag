"""Ask questions about the engineering dataset from your terminal."""
from src.application.rag_service import ask_question


def main():
    print("Engineering Knowledge RAG")
    print("Type a question, or type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            answer, chunks = ask_question(question)
        except Exception as error:
            print(f"Error: {error}\n")
            continue

        print("\n========== RAG ANSWER ==========")
        print(answer)

        # These are retrieved sources, not independently verified LLM citations.
        print("\nRetrieved sources:")
        seen = set()
        for doc in chunks:
            item = (doc.metadata.get("document_id"), doc.metadata.get("version"))
            if item not in seen:
                print(f"- {item[0]} (version {item[1]})")
                seen.add(item)
        print()


if __name__ == "__main__":
    main()
