"""Thin HTTP client for the backend API.

The UI talks to the FastAPI service over HTTP instead of importing the RAG logic
directly, so the two are fully decoupled - the UI knows nothing about ChromaDB,
the pipelines, or the chains. Point API_BASE_URL at a remote service to run the
UI against a deployed backend; it defaults to the local dev server.
"""
import os
from urllib.parse import quote

import requests

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 180  # OCR / vision ingestion can be slow


class APIError(Exception):
    """Raised when the API is unreachable or returns an error status."""


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{API_BASE}{path}"
    try:
        resp = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as e:
        raise APIError(
            f"Could not reach the API at {API_BASE}. "
            f"Is it running?  (uvicorn backend.api.main:app)\n\n{e}"
        )
    if not resp.ok:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        raise APIError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def list_documents() -> dict:
    """-> {"documents": [...], "chunk_count": int}"""
    return _request("GET", "/documents")


def query(question: str, scope, chat_history: list) -> dict:
    """-> {"answer": str, "sources": [str]}"""
    return _request("POST", "/query", json={
        "question": question,
        "scope": scope,
        "chat_history": chat_history,
    })


def ingest_file(file_bytes: bytes, filename: str, doc_type: str) -> dict:
    """-> {"doc_name", "chunks_added", "duplicate"}"""
    files = {"file": (filename, file_bytes)}
    data = {"doc_type": doc_type}
    return _request("POST", "/ingest/file", files=files, data=data)


def ingest_youtube(url: str) -> dict:
    """-> {"doc_name", "chunks_added", "duplicate"}"""
    return _request("POST", "/ingest/youtube", json={"url": url})


def delete_document(doc_name: str) -> dict:
    """-> {"doc_name", "chunks_deleted"}"""
    # Encode everything (spaces, parens, the colon in youtube:<id>) so the name
    # survives intact in the URL path; FastAPI decodes it back.
    return _request("DELETE", f"/documents/{quote(doc_name, safe='')}")


def clear() -> dict:
    """-> {"chunks_deleted": int}"""
    return _request("POST", "/clear")