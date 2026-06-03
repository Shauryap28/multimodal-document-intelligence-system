"""Streamlit UI - Multimodal Document Intelligence System.

Run from project root:
    streamlit run frontend/streamlit_app.py

Phase 2.5+:
  - Multi-provider LLM (Groq for text, Gemini reserved for Phase 3 vision)
  - YouTube transcript pipeline (URL input)
  - Image-of-text pipeline (.jpg / .png / etc. via EasyOCR)
  - Doc-type registry now carries per-entry file_types
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
from backend.services.pipelines.youtube import load_youtube_transcript

st.set_page_config(
    page_title="Multimodal Document Intelligence",
    page_icon="📄",
    layout="wide",
)

# --- Pipeline registry ---
# Each entry: loader fn + input kind ("file" or "url") + accepted file_types
# (only used for file inputs) + a short hint shown under the radio.
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
        "hint": "Photo or screenshot of typed text - book page, sign, document image. Uses EasyOCR.",
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


# --- Heavy resources cached across reruns ---
@st.cache_resource
def get_resources():
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    llm = get_llm("text")
    return embeddings, vectorstore, llm


embeddings, vectorstore, llm = get_resources()


# --- Sidebar: source picker + processor ---
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

            if doc_type_label == "Scanned PDF":
                spinner_msg = f"OCR'ing {uploaded.name}... this can take a while on CPU."
            elif doc_type_label == "Image (text)":
                spinner_msg = f"OCR'ing {uploaded.name}..."
            else:
                spinner_msg = f"Ingesting {uploaded.name}..."

            with st.spinner(spinner_msg):
                docs = config["loader"](save_path)
                if not docs:
                    st.warning("No text could be recovered from this file.")
                    st.stop()
                chunks = chunk_documents(docs)
                add_documents(vectorstore, chunks)

            st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")
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
                    chunks = chunk_documents(docs)
                    add_documents(vectorstore, chunks)
                except Exception as e:
                    st.error(f"Could not process video: {e}")
                    st.stop()

            st.success(f"Indexed {len(chunks)} chunks from the video transcript")
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.metric("Chunks in store", count(vectorstore))

    if st.button("Clear store"):
        clear(vectorstore)
        st.session_state.messages = []
        st.rerun()


# --- Main: chat ---
st.title("📄 Multimodal Document Intelligence")
st.caption(
    "Phase 2.5+ · PDFs + Scans + Images + YouTube · BGE-small · ChromaDB · Groq Llama 3.3 70B"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about your document, image, or video")
if prompt:
    if count(vectorstore) == 0:
        st.warning("Process a document, image, or video first.")
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