"""Turn a vector store into a retriever."""
from backend.config import settings


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})
