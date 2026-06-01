"""Streamlit UI - Multimodal Document Intelligence System.

Run from project root:
    streamlit run frontend/streamlit_app.py

Phase 2 update: doc-type radio (Text PDF / Scanned) acts as a manual router
between the two pipelines. Phase 4 will replace the manual pick with an
auto-detector that still allows override.
"""
import os
import sys
from pathlib import Path

# Make the project root importable when Streamlit launches us from frontend/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.rag.embeddings import get_embeddings
from backend.rag.chunking import chunk_documents
from backend.rag.vectorstore import get_vectorstore, add_documents, count, clear
from backend.rag.retriever import get_retriever
from backend.rag.chain import build_qa_chain
from backend.services.pipelines.text_pdf import load_text_pdf
from backend.services.pipelines.scanned import load_scanned_pdf

st.set_page_config(
    page_title="Multimodal Document Intelligence",
    page_icon="📄",
    layout="wide",
)

# --- Pipeline registry: dropdown label -> (loader fn, one-line hint) ---
DOC_TYPES = {
    "Text PDF": (
        load_text_pdf,
        "Selectable text - reports, papers, books. Tables handled automatically.",
    ),
    "Scanned": (
        load_scanned_pdf,
        "Scanned pages / no selectable text. Uses EasyOCR (slow, runs on CPU).",
    ),
}

# --- API-key sanity check ---
if not settings.GOOGLE_API_KEY:
    st.error("Set GOOGLE_API_KEY in your .env file before starting.")
    st.stop()


# --- Heavy resources cached across Streamlit reruns ---
@st.cache_resource
def get_resources():
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0,
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    )
    return embeddings, vectorstore, llm


embeddings, vectorstore, llm = get_resources()


# --- Sidebar: document management ---
with st.sidebar:
    st.header("Document")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    doc_type_label = st.radio(
        "What kind of PDF is this?",
        list(DOC_TYPES.keys()),
        help="This is a manual router for Phase 2. Auto-detect comes in Phase 4.",
    )
    st.caption(DOC_TYPES[doc_type_label][1])

    if uploaded is not None and st.button("Process document", type="primary"):
        os.makedirs("data/uploads", exist_ok=True)
        save_path = os.path.join("data/uploads", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        loader, _ = DOC_TYPES[doc_type_label]
        spinner_msg = (
            f"OCR'ing {uploaded.name}... this can take a while on CPU."
            if doc_type_label == "Scanned"
            else f"Ingesting {uploaded.name}..."
        )
        with st.spinner(spinner_msg):
            docs = loader(save_path)
            chunks = chunk_documents(docs)
            add_documents(vectorstore, chunks)

        st.success(f"Indexed {len(chunks)} chunks from {uploaded.name} ({doc_type_label})")
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.metric("Chunks in store", count(vectorstore))

    if st.button("Clear store"):
        clear(vectorstore)
        st.session_state.messages = []
        st.rerun()


# --- Main: title + chat ---
st.title("📄 Multimodal Document Intelligence")
st.caption(
    "Phase 2 · Text PDF + Scanned (EasyOCR) · BGE-small · ChromaDB · Gemini 2.5 Flash"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about your document")
if prompt:
    if count(vectorstore) == 0:
        st.warning("Upload and process a PDF first.")
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