"""FastAPI application entrypoint.

Run from the project root:
    uvicorn backend.api.main:app --reload

Interactive docs are then at http://127.0.0.1:8000/docs

The lifespan handler warms the heavy resources once at startup (embedding model,
Chroma client, LLM client) so the first request is not slow. Endpoints are sync
`def` on purpose: the underlying libraries (LLM SDK, embeddings, OCR) are
blocking, and FastAPI runs sync endpoints in a threadpool - writing `async def`
over blocking calls would stall the event loop.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api import deps
from backend.api.schemas import HealthResponse
from backend.api.routers import query, documents, ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    deps.warm_up()   # load embedding model + Chroma + LLM client once
    yield


app = FastAPI(
    title="Multimodal Document Intelligence API",
    version="0.8.0",
    lifespan=lifespan,
)

app.include_router(query.router)
app.include_router(documents.router)
app.include_router(ingest.router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok")