"""Persistent ChromaDB vector store with an HNSW cosine index.

Phase 4 adds list_documents() and delete_document(): the document list is
DERIVED from chunk metadata (the store is the single source of truth, so it
can never drift from what's actually indexed), and deletion can now target a
single document instead of wiping everything.
"""
from langchain_chroma import Chroma

from backend.config import settings


def get_vectorstore(embeddings):
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.PERSIST_DIR,
        collection_metadata={"hnsw:space": settings.DISTANCE_METRIC},
    )


def add_documents(vectorstore, documents):
    vectorstore.add_documents(documents)


def count(vectorstore) -> int:
    return vectorstore._collection.count()


def clear(vectorstore):
    """Remove ALL chunks from the store."""
    data = vectorstore.get()
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)


def list_documents(vectorstore) -> list[str]:
    """Distinct doc_name values currently in the store (for the query-scope picker).

    Derived from chunk metadata so it always matches what's actually indexed.
    Reads all metadata - fine at our scale; a very large store would keep a
    maintained registry instead.
    """
    data = vectorstore.get(include=["metadatas"])
    names = {
        m.get("doc_name")
        for m in data.get("metadatas", [])
        if m and m.get("doc_name")
    }
    return sorted(names)


def delete_document(vectorstore, doc_name: str) -> int:
    """Delete all chunks belonging to one document. Returns how many were removed."""
    data = vectorstore.get(where={"doc_name": doc_name})
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)