"""Local BGE-small embeddings (free, no API key)."""
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
