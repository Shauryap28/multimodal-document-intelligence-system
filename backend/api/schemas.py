"""Pydantic request/response models.

These define the API's contract: FastAPI validates incoming JSON against them,
serializes responses from them, and generates the interactive docs at /docs from
them. Keeping them in one place makes the contract easy to read and to evolve.
"""
from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """One turn of conversation, as sent by the client."""
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    # None or "" means "all documents"; otherwise a doc_name to scope retrieval.
    scope: str | None = None
    # Recent conversation turns; the server windows and converts these.
    chat_history: list[ChatTurn] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class DocumentsResponse(BaseModel):
    documents: list[str]
    chunk_count: int


class IngestResponse(BaseModel):
    doc_name: str
    chunks_added: int
    duplicate: bool  # True if the document was already indexed (nothing added)


class YoutubeIngestRequest(BaseModel):
    url: str


class DeleteResponse(BaseModel):
    doc_name: str
    chunks_deleted: int


class ClearResponse(BaseModel):
    chunks_deleted: int


class HealthResponse(BaseModel):
    status: str