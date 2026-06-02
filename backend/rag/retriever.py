"""Turn a vector store into a retriever.

Search strategy: MMR (Maximal Marginal Relevance).
MMR fetches a wider pool of candidates (fetch_k) and then re-picks top_k of
them by balancing relevance to the query against diversity from chunks already
picked. This prevents the retrieved context from being four paraphrases of the
same paragraph - especially valuable when a document repeats itself, when
chunks overlap heavily, or when a topic appears across multiple pages.
"""
from backend.config import settings


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.TOP_K,
            "fetch_k": settings.MMR_FETCH_K,
            "lambda_mult": settings.MMR_LAMBDA,
        },
    )