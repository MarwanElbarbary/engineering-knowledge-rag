"""Read the 24 main Markdown, TXT and PDF files and attach their metadata."""
import json

from langchain_community.document_loaders import TextLoader, PyPDFLoader

from src.config.setting import BASE_DIR, DOCUMENTS_DIR, METADATA_FILE


def load_documents():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    metadata_map = {
        item["document_id"]: item
        for item in metadata_list
    }
    all_documents = []

    # The 'uncleaned' dataset is intentionally excluded at this stage.
    for folder in ["markdown", "txt", "pdf"]:
        folder_path = DOCUMENTS_DIR / folder
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Missing documents folder: {folder_path}")

        for file_path in sorted(folder_path.iterdir()):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in [".md", ".txt"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                continue

            documents = loader.load()
            file_id = file_path.stem
            if file_id not in metadata_map:
                raise ValueError(f"Metadata not found for: {file_id}")

            for doc in documents:
                doc.metadata["source"] = str(file_path.relative_to(BASE_DIR))
                doc.metadata.update(metadata_map[file_id])

            all_documents.extend(documents)

    return all_documents
