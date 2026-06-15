"""Document management endpoints.

GET    /documents              - distinct document names (from chunk metadata,
                                 the single source of truth) + total chunk count.
DELETE /documents/{doc_name}   - remove one document's chunks.
POST   /clear                  - wipe the whole store.

doc_names can contain spaces, parentheses, or a colon (youtube:<id>); the client
URL-encodes them in the path, and FastAPI decodes them back.
"""
from fastapi import APIRouter, Depends, HTTPException

from backend.api import deps
from backend.api.schemas import DocumentsResponse, DeleteResponse, ClearResponse
from backend.rag.vectorstore import list_documents, count, delete_document, clear

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentsResponse)
def documents(vectorstore=Depends(deps.vectorstore)):
    return DocumentsResponse(
        documents=list_documents(vectorstore),
        chunk_count=count(vectorstore),
    )


@router.delete("/documents/{doc_name}", response_model=DeleteResponse)
def delete(doc_name: str, vectorstore=Depends(deps.vectorstore)):
    removed = delete_document(vectorstore, doc_name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{doc_name}'")
    return DeleteResponse(doc_name=doc_name, chunks_deleted=removed)


@router.post("/clear", response_model=ClearResponse)
def clear_store(vectorstore=Depends(deps.vectorstore)):
    before = count(vectorstore)
    clear(vectorstore)
    return ClearResponse(chunks_deleted=before)