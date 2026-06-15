"""Ingestion endpoints.

POST /ingest/file    - upload a file (PDF / image) plus a `doc_type` that selects
                       the pipeline. Mirrors the Streamlit flow: save the bytes to
                       data/uploads/, run the matching loader, chunk, and add.
POST /ingest/youtube - JSON {url}; fetches the transcript and indexes it.

Two duplicate-guard timings (same as the UI):
  - files:   the doc_name IS the filename, so we can check BEFORE the (expensive)
             loader runs.
  - youtube: the doc_name (youtube:<id>) is known only AFTER loading, so we check
             once the transcript is fetched.

A duplicate is reported as a normal 200 with duplicate=true and chunks_added=0,
so the client can show a friendly "already indexed" message.
"""
import os

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from backend.api import deps
from backend.api.schemas import IngestResponse, YoutubeIngestRequest
from backend.rag.chunking import chunk_documents
from backend.rag.vectorstore import add_documents, list_documents
from backend.services.pipelines.text_pdf import load_text_pdf
from backend.services.pipelines.scanned import load_scanned_pdf
from backend.services.pipelines.image_ocr import load_image
from backend.services.pipelines.image_vision import load_image_vision
from backend.services.pipelines.invoice import load_invoice
from backend.services.pipelines.youtube import load_youtube_transcript

router = APIRouter(tags=["ingest"])

UPLOAD_DIR = "data/uploads"

# API doc_type (snake_case, stable contract) -> loader, for file-based pipelines.
FILE_LOADERS = {
    "text_pdf": load_text_pdf,
    "scanned_pdf": load_scanned_pdf,
    "image_text": load_image,
    "visual": load_image_vision,
    "invoice": load_invoice,
}


def _ingest_docs(vectorstore, docs) -> int:
    """Chunk and add to the store; return the number of chunks added."""
    if not docs:
        return 0
    chunks = chunk_documents(docs)
    add_documents(vectorstore, chunks)
    return len(chunks)


@router.post("/ingest/file", response_model=IngestResponse)
def ingest_file(
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    vectorstore=Depends(deps.vectorstore),
):
    if doc_type not in FILE_LOADERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown doc_type '{doc_type}'. Expected one of: {list(FILE_LOADERS)}",
        )

    doc_name = file.filename
    # Duplicate guard up front: a file's doc_name is its filename.
    if doc_name in list_documents(vectorstore):
        return IngestResponse(doc_name=doc_name, chunks_added=0, duplicate=True)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, doc_name)
    with open(save_path, "wb") as f:
        f.write(file.file.read())

    try:
        docs = FILE_LOADERS[doc_type](save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process file: {e}")

    n = _ingest_docs(vectorstore, docs)
    return IngestResponse(doc_name=doc_name, chunks_added=n, duplicate=False)


@router.post("/ingest/youtube", response_model=IngestResponse)
def ingest_youtube(
    req: YoutubeIngestRequest,
    vectorstore=Depends(deps.vectorstore),
):
    try:
        docs = load_youtube_transcript(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process video: {e}")

    # doc_name (youtube:<id>) is known only after the transcript is fetched.
    doc_name = docs[0].metadata.get("doc_name") if docs else None
    if doc_name and doc_name in list_documents(vectorstore):
        return IngestResponse(doc_name=doc_name, chunks_added=0, duplicate=True)

    n = _ingest_docs(vectorstore, docs)
    return IngestResponse(doc_name=doc_name or "youtube", chunks_added=n, duplicate=False)