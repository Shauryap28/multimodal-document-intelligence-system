"""Streamlit UI - Multimodal Document Intelligence System.

Run from project root:
    streamlit run frontend/streamlit_app.py

Phase 3 (complete):
  - Image (visual)  -> Gemini Vision description (unstructured)
  - Invoice / Form  -> Gemini Vision + with_structured_output (structured fields)
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
from backend.rag.vectorstore import get_vectorstore, add_documents, count, clear
from backend.rag.retriever import get_retriever
from backend.rag.chain import build_qa_chain
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
        "loader": load_text_pdf,
        "input": "file",
        "file_types": ["pdf"],
        "hint": "Selectable text - reports, papers, books. Tables handled automatically.",
    },
    "Scanned PDF": {
        "loader": load_scanned_pdf,
        "input": "file",
        "file_types": ["pdf"],
        "hint": "Scanned pages / no selectable text. Uses EasyOCR (slow, runs on CPU).",
    },
    "Image (text)": {
        "loader": load_image,
        "input": "file",
        "file_types": ["png", "jpg", "jpeg", "webp", "bmp"],
        "hint": "Photo/screenshot of typed text. Uses EasyOCR (free, local).",
    },
    "Visual / Handwritten": {
        "loader": load_image_vision,
        "input": "file",
        "file_types": ["pdf", "png", "jpg", "jpeg", "webp", "bmp"],
        "hint": "Charts, diagrams, handwriting, photos, complex/handwritten PDFs. Gemini Vision (1 call per page).",
    },
    "Invoice / Form": {
        "loader": load_invoice,
        "input": "file",
        "file_types": ["pdf", "png", "jpg", "jpeg", "webp"],
        "hint": "Receipts, bills, forms. Gemini Vision extracts structured fields (vendor, total, items).",
    },
    "YouTube": {
        "loader": load_youtube_transcript,
        "input": "url",
        "hint": "Paste a YouTube URL. Uses the video's auto-generated transcript.",
    },
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


def ingest(docs):
    if not docs:
        return 0
    chunks = chunk_documents(docs)
    add_documents(vectorstore, chunks)
    return len(chunks)


# Spinner copy per pipeline
_SPINNERS = {
    "Scanned PDF": "OCR'ing {name}...",
    "Image (text)": "OCR'ing {name}...",
    "Visual / Handwritten": "Asking Gemini Vision to read {name}...",
    "Invoice / Form": "Extracting structured fields from {name} with Gemini Vision...",
}


with st.sidebar:
    st.header("Document")

    doc_type_label = st.radio(
        "What kind of input?",
        list(DOC_TYPES.keys()),
        help="Manual router. Auto-detect comes in Phase 4.",
    )
    config = DOC_TYPES[doc_type_label]
    st.caption(config["hint"])

    if config["input"] == "file":
        uploaded = st.file_uploader("Upload file", type=config["file_types"])
        if uploaded is not None and st.button("Process", type="primary"):
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
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        if url and st.button("Process video", type="primary"):
            with st.spinner("Fetching transcript and indexing..."):
                try:
                    docs = config["loader"](url)
                except Exception as e:
                    st.error(f"Could not process video: {e}")
                    st.stop()
                n = ingest(docs)
            st.success(f"Indexed {n} chunks from the video transcript")
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.metric("Chunks in store", count(vectorstore))

    if st.button("Clear store"):
        clear(vectorstore)
        st.session_state.messages = []
        st.rerun()


st.title("📄 Multimodal Document Intelligence")
st.caption(
    "Phase 3 · PDFs + Scans + Images (OCR/Vision) + Invoices + YouTube · "
    "BGE-small · ChromaDB · Groq + Gemini"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about your document, image, invoice, or video")
if prompt:
    if count(vectorstore) == 0:
        st.warning("Process something first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retriever = get_retriever(vectorstore)
        chain = build_qa_chain(retriever, llm)
        with st.spinner("Thinking..."):
            answer = chain.invoke(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})