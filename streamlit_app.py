import streamlit as st
from src.application.data_browser import knowledge_base_page
from src.application.rag_service import ask_question
from src.config.setting import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_MODEL,
    LLM_MODEL
)


from src.retrivar.vector_store import (
    get_vector_store,
    build_vector_database
)


st.set_page_config(
    page_title="Engineering Knowledge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner=False)
def initialize_knowledge_base():

    vector_store = get_vector_store()

    # Build the index only if it is empty
    if not vector_store.get(limit=1)["ids"]:
        build_vector_database()

    return vector_store


with st.spinner("Preparing the knowledge base..."):
    initialize_knowledge_base()


st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #101a35, #223b66);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 22px;
    color: white;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #cbd5e1;
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("🧠 Engineering AI")
    st.caption("Retrieval-Augmented Generation")

    page = st.radio(
        "Navigation",
        [
            "💬 Chatbot",
            "📚 Knowledge Base",
            "📖 About & Architecture"
        ]
    )

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def chat_page():

    st.markdown("""
    <div class="hero">
        <h1>Engineering Knowledge Assistant</h1>
        <p>
            Ask questions about engineering architecture,
            incidents, APIs, operational runbooks,
            and technical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Clear conversation
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 Retrieved Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['document_id']}** "
                            f"— Version {source['version']}"
                        )
                        st.caption(source["path"])

    # User input
    question = st.chat_input(
        "Ask a question about the engineering knowledge base..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching the knowledge base..."):

                try:
                    answer, chunks = ask_question(question)

                    st.markdown(answer)

                    # Extract unique sources
                    sources = []
                    seen = set()

                    for doc in chunks:
                        metadata = doc.metadata

                        key = (
                            metadata.get("document_id"),
                            metadata.get("version")
                        )

                        if key in seen:
                            continue

                        seen.add(key)

                        sources.append({
                            "document_id": metadata.get(
                                "document_id", "Unknown"
                            ),
                            "version": metadata.get(
                                "version", "Unknown"
                            ),
                            "path": metadata.get(
                                "source", "Unknown"
                            )
                        })

                    if sources:
                        with st.expander("📚 Retrieved Sources"):
                            for source in sources:
                                st.markdown(
                                    f"**{source['document_id']}** "
                                    f"— Version {source['version']}"
                                )
                                st.caption(source["path"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception:
                    st.error(
                        "Unable to generate an answer. "
                        "Please check the RAG service and API connection."
                    )


def about_page():

    st.title("📖 About the Engineering Knowledge RAG")
    st.markdown(
        "An AI-powered knowledge assistant built using "
        "Retrieval-Augmented Generation (RAG)."
    )

    st.subheader("🎯 Project Overview")

    st.write("""
    This project is designed to answer engineering-related
    questions using an internal technical knowledge base.

    The dataset includes architecture decisions, API
    documentation, incident reports, operational runbooks,
    and engineering guides.

    Instead of relying only on an LLM's general knowledge,
    the system retrieves relevant information from the
    documents and uses it to generate grounded answers.
    """)

    st.subheader("📊 Knowledge Base")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Documents", "24")
    col2.metric("Indexed Chunks", "26")
    col3.metric("Embedding Dimensions", "384")
    col4.metric("Top-K Retrieval", TOP_K)

    st.caption(
        "These figures describe the current baseline dataset "
        "and indexing configuration."
    )

    st.divider()

    st.subheader("⚙️ RAG Architecture")

    st.markdown("#### Phase 1 — Document Indexing")

    st.code("""
Documents (PDF, Markdown, TXT)
              |
              v
        Document Loading
              |
              v
       Metadata Enrichment
              |
              v
        Text Chunking
              |
              v
       Embedding Model
              |
              v
       Chroma Vector DB
""", language="text")

    st.markdown("#### Phase 2 — Question Answering")

    st.code("""
         User Question
               |
               v
        Query Embedding
               |
               v
        Semantic Retrieval
               |
               v
      Top-K Relevant Chunks
               |
               v
          Prompt Builder
               |
               v
            Groq LLM
               |
               v
      Answer + Source IDs
""", language="text")

    st.divider()

    st.subheader("🔧 Technical Implementation")

    st.markdown(f"""
    **1. Document Ingestion**

    Load PDF, Markdown, and TXT documents into
    LangChain Document objects with metadata.

    **2. Text Chunking**

    Use RecursiveCharacterTextSplitter.

    - Chunk Size: `{CHUNK_SIZE}` characters
    - Chunk Overlap: `{CHUNK_OVERLAP}` characters

    **3. Embedding Generation**

    Model: `{EMBEDDING_MODEL}`

    Generates 384-dimensional vector representations.

    **4. Vector Storage**

    Store document chunks, embeddings, and metadata
    in a persistent Chroma database.

    **5. Semantic Retrieval**

    Retrieve the top `{TOP_K}` most semantically similar
    chunks to the user query.

    **6. Context Augmentation**

    Combine the retrieved context, document identifiers,
    versions, and the user question into a structured prompt.

    **7. Answer Generation**

    LLM: `{LLM_MODEL}` via Groq.

    Generate answers grounded in the retrieved context.
    """)

    st.divider()

    st.subheader("🛡️ Reliability & Limitations")

    st.write("""
    The assistant is instructed to use the retrieved
    documents, prefer current accepted decisions over
    superseded ones, and acknowledge missing information.

    The current implementation uses semantic similarity
    retrieval. It does not yet include reranking, hybrid
    search, conversational memory, or automated evaluation.

    The additional uncleaned dataset is retained for
    future work but is not included in the current index.
    """)

    st.info(
        "Next planned step: Evaluate retrieval quality "
        "using the dataset's prepared evaluation questions."
    )


if __name__ == "__main__":
    if page == "💬 Chatbot":
        chat_page()
    elif page == "📚 Knowledge Base":
        knowledge_base_page()
    else:
        about_page()