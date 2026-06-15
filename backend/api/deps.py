"""Shared API resources, created once and cached.

The heavy objects - the embedding model, the Chroma client, the LLM client -
are expensive to build, so they are created lazily and memoized with
`lru_cache`. The app warms them at startup via `warm_up()` so the first request
is not slow.

This mirrors Streamlit's `@st.cache_resource`, but now the API is the single
owner of these resources (and of the ChromaDB on disk). Endpoints receive them
through FastAPI's `Depends`, which makes them easy to override in tests.
"""
from functools import lru_cache

from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import get_vectorstore
from backend.rag.llm import get_llm


@lru_cache(maxsize=1)
def embeddings():
    return get_embeddings()


@lru_cache(maxsize=1)
def vectorstore():
    return get_vectorstore(embeddings())


@lru_cache(maxsize=1)
def text_llm():
    return get_llm("text")


def warm_up():
    """Eagerly initialize the heavy resources (called once at startup)."""
    vectorstore()
    text_llm()