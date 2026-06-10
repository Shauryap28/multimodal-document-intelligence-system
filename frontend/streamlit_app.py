"""Streamlit UI - Multimodal Document Intelligence System.

Run from project root:
    streamlit run frontend/streamlit_app.py

Phase 4: per-document query scoping + per-document delete + duplicate-ingestion
guard (a document already in the store is not silently indexed a second time).
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st

from backend.config import settings
from backend.rag.embeddings import get_embeddings
from backend.rag.chunking import chunk_documents
from backend.rag.vectorstore import (
    get_vectorstore, add_documents, count, clear,
    list_documents, delete_document,
)
from backend.rag.retriever import get_retriever
from backend.rag.chain import build_qa_chain_with_sources, format_sources
from backend.rag.llm import get_llm
from backend.services.pipelines.text_pdf import load_text_pdf
from backend.services.pipelines.scanned import load_scanned_pdf
from backend.services.pipelines.image_ocr import load_image
from backend.services.pipelines.image_vision import load_image_vision
from backend.services.pipelines.invoice import load_invoice
from backend.services.pipelines.youtube import load_youtube_transcript

st.set_page_config(
    page_title="Multimodal Document Intelligence",
    page_icon="📄",
    layout="wide",
)

# --- Pipeline registry ---
DOC_TYPES = {
    "Text PDF": {
        "loader": load_text_pdf, "input": "file", "file_types": ["pdf"],
        "hint": "Selectable text - reports, papers, books. Tables handled automatically.",
    },
    "Scanned PDF": {
        "loader": load_scanned_pdf, "input": "file", "file_types": ["pdf"],
        "hint": "Scanned pages / no selectable text. Uses EasyOCR (slow, runs on CPU).",
    },
    "Image (text)": {
        "loader": load_image, "input": "file",
        "file_types": ["png", "jpg", "jpeg", "webp", "bmp"],
        "hint": "Photo/screenshot of typed text. Uses EasyOCR (free, local).",
    },
    "Visual / Handwritten": {
        "loader": load_image_vision, "input": "file",
        "file_types": ["pdf", "png", "jpg", "jpeg", "webp", "bmp"],
        "hint": "Charts, diagrams, handwriting, photos, complex/handwritten PDFs. Gemini Vision (1 call per page).",
    },
    "Invoice / Form": {
        "loader": load_invoice, "input": "file",
        "file_types": ["pdf", "png", "jpg", "jpeg", "webp"],
        "hint": "Receipts, bills, forms. Gemini Vision extracts structured fields (vendor, total, items).",
    },
    "YouTube": {
        "loader": load_youtube_transcript, "input": "url",
        "hint": "Paste a YouTube URL. Uses the video's auto-generated transcript.",
    },
}

_SPINNERS = {
    "Scanned PDF": "OCR'ing {name}...",
    "Image (text)": "OCR'ing {name}...",
    "Visual / Handwritten": "Asking Gemini Vision to read {name}...",
    "Invoice / Form": "Extracting structured fields from {name} with Gemini Vision...",
}

# --- API key check ---
if not settings.GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
        "and add it to your .env file as GROQ_API_KEY=..."
    )
    st.stop()


@st.cache_resource
def get_resources():
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    llm = get_llm("text")
    return embeddings, vectorstore, llm


embeddings, vectorstore, llm = get_resources()
docs_in_store = list_documents(vectorstore)


def ingest(docs):
    if not docs:
        return 0
    chunks = chunk_documents(docs)
    add_documents(vectorstore, chunks)
    return len(chunks)


# --- Sidebar ---
with st.sidebar:
    st.header("Add a document")
    doc_type_label = st.radio(
        "What kind of input?",
        list(DOC_TYPES.keys()),
        help="Manual router. Auto-detect is a future enhancement.",
    )
    config = DOC_TYPES[doc_type_label]
    st.caption(config["hint"])

    if config["input"] == "file":
        uploaded = st.file_uploader("Upload file", type=config["file_types"])
        if uploaded is not None and st.button("Process", type="primary"):
            # Duplicate guard (cheap, before the expensive loader runs):
            # a file's doc_name is its filename, so we can check up front.
            if uploaded.name in docs_in_store:
                st.warning(
                    f"'{uploaded.name}' is already indexed. "
                    "Delete it in Manage first if you want to re-ingest."
                )
                st.stop()

            os.makedirs("data/uploads", exist_ok=True)
            save_path = os.path.join("data/uploads", uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            spinner_msg = _SPINNERS.get(doc_type_label, "Ingesting {name}...").format(
                name=uploaded.name
            )
            with st.spinner(spinner_msg):
                try:
                    docs = config["loader"](save_path)
                except Exception as e:
                    st.error(f"Could not process file: {e}")
                    st.stop()
                n = ingest(docs)
            if n == 0:
                st.warning("No usable content could be recovered from this file.")
            else:
                st.success(f"Indexed {n} chunks from {uploaded.name}")
                st.session_state.messages = []
                st.rerun()

    elif config["input"] == "url":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if url and st.button("Process video", type="primary"):
            with st.spinner("Fetching transcript and indexing..."):
                try:
                    docs = config["loader"](url)
                except Exception as e:
                    st.error(f"Could not process video: {e}")
                    st.stop()
                # Duplicate guard: the video's doc_name (youtube:<id>) is known
                # only after loading, so we check here, before adding.
                doc_name = docs[0].metadata.get("doc_name") if docs else None
                if doc_name and doc_name in docs_in_store:
                    st.warning(f"This video ({doc_name}) is already indexed.")
                    st.stop()
                n = ingest(docs)
            st.success(f"Indexed {n} chunks from the video transcript")
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.header("Manage")
    st.metric("Chunks in store", count(vectorstore))
    st.caption(f"{len(docs_in_store)} document(s) indexed")

    if docs_in_store:
        to_delete = st.selectbox("Delete a document", ["-"] + docs_in_store)
        if to_delete != "-" and st.button("Delete selected"):
            removed = delete_document(vectorstore, to_delete)
            st.success(f"Deleted {removed} chunks from {to_delete}")
            st.session_state.messages = []
            st.rerun()

    if st.button("Clear entire store"):
        clear(vectorstore)
        st.session_state.messages = []
        st.rerun()


# --- Main ---
st.title("📄 Multimodal Document Intelligence")
st.caption(
    "Phase 6 · answers with sources · per-document scoping · PDFs + Scans + "
    "Images + Invoices + YouTube · BGE-small · ChromaDB · Groq + Gemini"
)

scope = st.selectbox(
    "Ask about",
    ["All documents"] + docs_in_store,
    help="Scope the question to one document to avoid interference from others.",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about your document, image, invoice, or video")
if prompt:
    if count(vectorstore) == 0:
        st.warning("Process something first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        doc_filter = None if scope == "All documents" else scope
        retriever = get_retriever(vectorstore, doc_name=doc_filter)
        chain = build_qa_chain_with_sources(retriever, llm)
        with st.spinner("Thinking..."):
            result = chain.invoke(prompt)
        answer = result["answer"]
        sources = format_sources(result["docs"])
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )