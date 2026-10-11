
import json
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from src.config.setting import BASE_DIR


DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"


# Read Markdown, TXT, HTML and PDF
@st.cache_data(show_spinner=False)
def read_document(file_path, modified_time):

    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        text = "\n\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        return text

    return path.read_text(
        encoding="utf-8",
        errors="replace"
    )


# Discover available files
def get_files(folders):

    files = []

    for folder_name in folders:
        folder = DOCS_DIR / folder_name

        if folder.exists():
            files.extend(
                file for file in folder.iterdir()
                if file.is_file()
            )

    return sorted(files, key=lambda x: x.name.lower())


# Reusable document viewer
def document_viewer(files, key_prefix, metadata_map=None):

    search = st.text_input(
        "🔎 Search by filename or content",
        key=f"{key_prefix}_search"
    ).strip().lower()

    if search:
        filtered = []

        for file in files:
            content = read_document(
                str(file),
                file.stat().st_mtime
            )

            title = ""
            if metadata_map:
                title = metadata_map.get(
                    file.stem, {}
                ).get("title", "")

            if (
                search in file.name.lower()
                or search in title.lower()
                or search in content.lower()
            ):
                filtered.append(file)

        files = filtered

    if not files:
        st.info("No matching documents found.")
        return

    selected = st.selectbox(
        "Select a document",
        files,
        format_func=lambda f: f.name,
        key=f"{key_prefix}_select"
    )

    st.caption(
        f"File: {selected.relative_to(BASE_DIR)}"
    )

    # Show metadata when available
    if metadata_map:
        metadata = metadata_map.get(
            selected.stem, {}
        )

        if metadata:
            st.markdown(
                f"**Title:** {metadata.get('title', 'N/A')}"
            )

            st.markdown(
                f"**Version:** {metadata.get('version', 'N/A')}"
            )

            with st.expander("Document Metadata"):
                st.json(metadata)

    content = read_document(
        str(selected),
        selected.stat().st_mtime
    )

    st.markdown("### Document Content")

    if selected.suffix.lower() == ".md":
        st.markdown(content)

    elif selected.suffix.lower() == ".html":
        st.code(content, language="html")

    else:
        st.text_area(
            "Extracted text" if selected.suffix.lower() == ".pdf"
            else "File content",
            value=content or "No extractable text found.",
            height=400,
            disabled=True,
            key=f"{key_prefix}_{selected.name}"
        )

    st.download_button(
        "⬇️ Download Original Document",
        data=selected.read_bytes(),
        file_name=selected.name,
        key=f"{key_prefix}_download"
    )


# Main Knowledge Base page
def knowledge_base_page():

    st.title("📚 Engineering Knowledge Base")

    st.write("""
    Explore the technical documents powering the
    Engineering Knowledge RAG Assistant.

    Browse architecture decisions, operational runbooks,
    engineering guides, API documentation and incidents.
    """)

    indexed_files = get_files(
        ["markdown", "pdf", "txt"]
    )

    raw_files = get_files(["uncleaned"])

    with open(
        DATA_DIR / "metadata" / "documents.json",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    metadata_map = {
        item["document_id"]: item
        for item in metadata
    }

    with open(
        DATA_DIR / "evaluation" / "questions.json",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)

    # Overview metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Indexed Documents", len(indexed_files))
    col2.metric("Raw Documents", len(raw_files))
    col3.metric("Evaluation Questions", len(questions))

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📄 Indexed Documents",
        "🗂️ Raw Documents",
        "📊 Metadata & Evaluation"
    ])

    # Indexed documents
    with tab1:
        st.subheader("Indexed Knowledge Documents")

        st.info(
            "These documents form the current RAG knowledge base."
        )

        document_viewer(
            indexed_files,
            "indexed",
            metadata_map
        )

    # Raw documents
    with tab2:
        st.subheader("Raw / Uncleaned Documents")

        st.warning(
            "These files are available for exploration "
            "but are not included in the current vector index."
        )

        document_viewer(
            raw_files,
            "raw"
        )

    # Metadata and evaluation
    with tab3:
        st.subheader("Document Metadata")

        st.dataframe(
            metadata,
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        st.subheader("Evaluation Questions")

        evaluation_rows = [
            {
                "ID": item["id"],
                "Question": item["question"],
                "Expected Sources": ", ".join(
                    item["expected_sources"]
                ),
                "Type": item.get("type", "")
            }
            for item in questions
        ]

        st.dataframe(
            evaluation_rows,
            hide_index=True,
            use_container_width=True
        )

        manifest_path = (
            DATA_DIR / "metadata" / "uncleaned_manifest.json"
        )

        if manifest_path.exists():
            with st.expander("Raw Dataset Manifest"):
                with open(
                    manifest_path,
                    encoding="utf-8"
                ) as f:
                    st.json(json.load(f))

        readme_path = DATA_DIR / "README.md"

        if readme_path.exists():
            with st.expander("Dataset README"):
                st.markdown(
                    readme_path.read_text(encoding="utf-8")
                )
