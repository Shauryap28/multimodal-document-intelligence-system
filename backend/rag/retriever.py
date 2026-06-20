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


def get_hybrid_retriever(vectorstore, doc_name: str | None = None):
    """Dense (Chroma) + BM25 keyword search, fused via reciprocal rank fusion.

    Both channels search the SAME pool independently - the document-scoped set of
    chunks - and their ranked lists are merged. BM25 does NOT search the dense
    retriever's output; it searches the whole pool, which is exactly how it can
    surface an exact-token chunk (a SKU, an email, a code) that dense missed.

    BM25Retriever has no metadata filter, so scoping is done by building it over
    only the scoped document's chunks, pulled from Chroma. Rebuilt per call -
    cheap at this scale, and never out of sync with the store.
    """
    # imported lazily so the rest of the app doesn't require rank_bm25.
    # EnsembleRetriever moved to langchain_classic in newer LangChain versions.
    #try:
    from langchain_classic.retrievers import EnsembleRetriever
    #except ImportError:
        #from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    dense = get_retriever(vectorstore, doc_name=doc_name)

    # The scoped chunk pool, fetched straight from Chroma.
    data = vectorstore.get(where={"doc_name": doc_name}) if doc_name else vectorstore.get()
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(data["documents"], data["metadatas"])
    ]
    if not docs:
        return dense   # empty pool -> nothing for BM25 to index

    sparse = BM25Retriever.from_documents(docs)
    sparse.k = settings.TOP_K

    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=settings.HYBRID_WEIGHTS,
        c=settings.HYBRID_RRF_C,
    )