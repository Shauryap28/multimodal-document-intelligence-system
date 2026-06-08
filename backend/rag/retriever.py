"""MMR retriever, optionally scoped to a single document via a metadata filter.

When doc_name is given, ChromaDB applies it as a `where` clause DURING the
search (pre-filtering), so MMR diversifies only among that document's chunks.
This fixes the multi-document interference measured in testing: with several
documents in the store, an unfiltered search could fill the k slots with chunks
from unrelated documents.
"""
from backend.config import settings


def get_retriever(vectorstore, doc_name: str | None = None):
    search_kwargs = {
        "k": settings.TOP_K,
        "fetch_k": settings.MMR_FETCH_K,
        "lambda_mult": settings.MMR_LAMBDA,
    }
    if doc_name:
        search_kwargs["filter"] = {"doc_name": doc_name}
    return vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)