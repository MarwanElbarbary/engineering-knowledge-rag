# Engineering Knowledge RAG — Our From-Scratch Project

A small RAG system built step by step with **Python + LangChain + Hugging Face embeddings + Chroma + Groq**. This is a **new implementation** using only the dataset from the supplied repository. No old source code, notebook, API key or vector database is copied into this project.

## Project layout

```text
my-rag-project/
├── data/
│   ├── documents/
│   │   ├── markdown/      # 17 main files
│   │   ├── pdf/           # 5 main files
│   │   ├── txt/           # 2 main files
│   │   └── uncleaned/     # 10 raw files kept for later (NOT indexed now)
│   ├── metadata/
│   │   ├── documents.json
│   │   └── uncleaned_manifest.json
│   ├── evaluation/
│   │   └── questions.json # 30 evaluation questions; NOT indexed
│   └── chroma_db/        # created after indexing; not included in ZIP
├── src/
│   ├── application/
│   │   └── rag_service.py # joins retrieval, prompt and LLM
│   ├── config/
│   │   └── setting.py     # paths, model names, settings
│   ├── generation/
│   │   ├── llm.py         # Groq ChatGroq client
│   │   └── prompt.py      # instructions + context + question
│   ├── ingestion/
│   │   ├── loader.py      # 24 docs + document metadata
│   │   └── chunker.py     # chunk_size=600, overlap=100
│   └── retrivar/         # spelling matches the supplied repo screenshot
│       ├── embeddings.py # all-MiniLM-L6-v2
│       ├── retriever.py  # similarity search with k=3
│       └── vector_store.py # Chroma DB build/open
├── main.py               # interactive terminal questions
├── requirements.txt
├── .env.example
└── .gitignore
```

## Run in VS Code (Windows PowerShell)

Open the terminal **inside the `my-rag-project` folder**.

1. Create a Python 3.12 virtual environment (only once):

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On a new terminal, activate `.venv` again. Check with `python --version`.

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Create `.env` using `.env.example`. Add **your own fresh** `GROQ_API_KEY`. Never publish `.env` or paste the key into chat.

4. **Build the database** from the main 24 documents:

   ```powershell
   python -m src.retrivar.vector_store
   ```

   On this dataset and our current chunk settings the expected output is **24 LangChain Documents and 26 Chunks**. Chroma is saved at `data/chroma_db/`. The first run may download the embedding model from Hugging Face.

5. **Ask questions**:

   ```powershell
   python main.py
   ```

   Try:
   - `Is Kafka still the default for all asynchronous communication?`
   - `Why does Kafka support event replay?`
   - `What is the CEO's favourite food?`

   Type `exit` to stop. Groq access requires an internet connection and an enabled API key.

## How the parts connect

**Build once:** document loader -> metadata -> chunker -> embedding model -> Chroma DB.

**For every question:** user question -> same embedding model -> Chroma similarity search (`k=3`) -> prompt (question + retrieved text + document IDs / versions) -> Groq LLM -> answer.

## What this version includes (and does not)

- Same working setup as our experiments: `RecursiveCharacterTextSplitter(600, 100)`, `all-MiniLM-L6-v2`, Chroma, `k=3`, `ChatGroq(model="openai/gpt-oss-20b", temperature=0, reasoning_format="hidden")`, and our original prompt instructions.
- Included the clean dataset, the raw `uncleaned` materials and the 30 evaluation questions **as files**. Only the clean 24 documents enter the vector database for now.
- **Not added yet:** evaluation automation, memory, hybrid search, reranking, API or UI. We can add these **after** validating the organized project.
- Retrieval uses semantic similarity; it can return superseded material. The prompt asks the model to prefer the accepted version, but we have not implemented strict version filtering yet.
- Documents that do not answer a question may still be retrieved; the model is instructed to say `I don't know` without evidence. This is not a guarantee.
- `Retrieved sources` in the terminal are the chunks returned by the retriever, not a formal verification of the LLM's citations.
- If you later change the chunk settings or source dataset, delete `data/chroma_db/` and rebuild to avoid stale chunk records. Rebuilding the same dataset uses stable IDs.

**Security:** No secret key is included. Do not copy `.env` or any secrets from the older repository.
