"""Persistent ChromaDB vector store.

Chroma builds an HNSW (approximate-nearest-neighbour) index automatically
as documents are added. We set the distance metric to cosine to match the
normalized BGE embeddings. With persist_directory set, vectors are written
to disk and reloaded on the next run - so we embed each document only once.
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import settings


def get_vectorstore(embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.PERSIST_DIR,
        collection_metadata={"hnsw:space": settings.DISTANCE_METRIC},
    )


def add_documents(vectorstore: Chroma, chunks: list[Document]) -> None:
    vectorstore.add_documents(chunks)


def count(vectorstore: Chroma) -> int:
    """Number of chunks currently stored (0 == empty / not yet ingested)."""
    return vectorstore._collection.count()

def clear(vectorstore: Chroma) -> None:
    """Delete every chunk from the store (used by the UI's 'Clear store' button)."""
    ids = vectorstore.get()["ids"]
    if ids:
        vectorstore.delete(ids=ids)