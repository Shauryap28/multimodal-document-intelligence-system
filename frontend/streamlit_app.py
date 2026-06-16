"""Streamlit UI - Multimodal Document Intelligence System.

Run from project root (the API must be running too):
    uvicorn backend.api.main:app --reload          # terminal 1
    streamlit run frontend/streamlit_app.py        # terminal 2

Phase 8: the UI is a THIN CLIENT. It owns no RAG logic, no ChromaDB, no models -
it talks to the FastAPI backend over HTTP via api_client. Ingestion, querying,
listing, deletion, and clearing are all API calls. This decouples the UI from
the backend: either can be swapped or deployed independently.
"""
import streamlit as st

import api_client
from api_client import APIError

st.set_page_config(page_title="Multimodal Document Intelligence", page_icon="📄")

# --- Input-type registry (display only; loaders live server-side) ---
# `api_type` matches the backend FILE_LOADERS keys; YouTube uses its own endpoint.
DOC_TYPES = {
    "Text PDF": {
        "api_type": "text_pdf", "input": "file", "file_types": ["pdf"],
        "hint": "Selectable text - reports, papers, books. Tables handled automatically.",
    },
    "Scanned PDF": {
        "api_type": "scanned_pdf", "input": "file", "file_types": ["pdf"],
        "hint": "Scanned pages / no selectable text. Uses EasyOCR (slow, runs on CPU).",
    },
    "Image (text)": {
        "api_type": "image_text", "input": "file",
        "file_types": ["png", "jpg", "jpeg", "webp", "bmp"],
        "hint": "Photo/screenshot of typed text. Uses EasyOCR (free, local).",
    },
    "Visual / Handwritten": {
        "api_type": "visual", "input": "file",
        "file_types": ["pdf", "png", "jpg", "jpeg", "webp", "bmp"],
        "hint": "Charts, diagrams, handwriting, photos, complex/handwritten PDFs. Gemini Vision (1 call per page).",
    },
    "Invoice / Form": {
        "api_type": "invoice", "input": "file",
        "file_types": ["pdf", "png", "jpg", "jpeg", "webp"],
        "hint": "Receipts, bills, forms. Gemini Vision extracts structured fields (vendor, total, items).",
    },
    "YouTube": {
        "input": "url",
        "hint": "Paste a YouTube URL. Uses the video's auto-generated transcript.",
    },
}

_SPINNERS = {
    "Scanned PDF": "OCR'ing {name}...",
    "Image (text)": "OCR'ing {name}...",
    "Visual / Handwritten": "Asking Gemini Vision to read {name}...",
    "Invoice / Form": "Extracting structured fields from {name} with Gemini Vision...",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Fetch current store state from the API (and verify it is reachable) ---
try:
    store = api_client.list_documents()
except APIError as e:
    st.error(str(e))
    st.stop()

docs_in_store = store["documents"]
chunk_count = store["chunk_count"]

# --- Sidebar: add + manage ---
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
            spinner_msg = _SPINNERS.get(doc_type_label, "Ingesting {name}...").format(
                name=uploaded.name
            )
            with st.spinner(spinner_msg):
                try:
                    result = api_client.ingest_file(
                        uploaded.getvalue(), uploaded.name, config["api_type"]
                    )
                except APIError as e:
                    st.error(str(e))
                    st.stop()
            if result["duplicate"]:
                st.warning(
                    f"'{result['doc_name']}' is already indexed. "
                    "Delete it in Manage first if you want to re-ingest."
                )
            elif result["chunks_added"] == 0:
                st.warning("No usable content could be recovered from this file.")
            else:
                st.success(f"Indexed {result['chunks_added']} chunks from {result['doc_name']}")
                st.session_state.messages = []
                st.rerun()

    elif config["input"] == "url":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if url and st.button("Process video", type="primary"):
            with st.spinner("Fetching transcript and indexing..."):
                try:
                    result = api_client.ingest_youtube(url)
                except APIError as e:
                    st.error(str(e))
                    st.stop()
            if result["duplicate"]:
                st.warning(f"This video ({result['doc_name']}) is already indexed.")
            elif result["chunks_added"] == 0:
                st.warning("No transcript content could be recovered.")
            else:
                st.success(f"Indexed {result['chunks_added']} chunks from the video transcript")
                st.session_state.messages = []
                st.rerun()

    st.divider()
    st.header("Manage")
    st.metric("Chunks in store", chunk_count)
    st.caption(f"{len(docs_in_store)} document(s) indexed")

    if docs_in_store:
        to_delete = st.selectbox("Delete a document", ["-"] + docs_in_store)
        if to_delete != "-" and st.button("Delete selected"):
            try:
                result = api_client.delete_document(to_delete)
            except APIError as e:
                st.error(str(e))
                st.stop()
            st.success(f"Deleted {result['chunks_deleted']} chunks from {result['doc_name']}")
            st.session_state.messages = []
            st.rerun()

    if st.button("Clear entire store"):
        try:
            api_client.clear()
        except APIError as e:
            st.error(str(e))
            st.stop()
        st.session_state.messages = []
        st.rerun()


# --- Main ---
st.title("📄 Multimodal Document Intelligence")
st.caption(
    "Phase 8 · FastAPI backend · conversation memory · answers with sources · "
    "per-document scoping · PDFs + Scans + Images + Invoices + YouTube · "
    "BGE-small · ChromaDB · Groq + Gemini"
)

scope_label = st.selectbox(
    "Ask about",
    ["All documents"] + docs_in_store,
    help="Scope the question to one document to avoid interference from others.",
)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about your document, image, invoice, or video")
if prompt:
    if chunk_count == 0:
        st.warning("Process something first.")
        st.stop()

    # Prior turns become chat_history (dialogue only); the server windows it.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        scope = None if scope_label == "All documents" else scope_label
        with st.spinner("Thinking..."):
            try:
                result = api_client.query(prompt, scope, history)
            except APIError as e:
                st.error(str(e))
                st.stop()
        answer = result["answer"]
        sources = result["sources"]
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )