"""Text-PDF pipeline.

Two extractors working together:
  - PyMuPDF (via PyMuPDFLoader) for the page text
  - pdfplumber for tables, which plain text extraction mangles

Every Document carries the project's metadata schema:
  doc_name, doc_type, page_number, content_type, upload_time
(chunk_index is added later, during chunking).
"""
import os
from datetime import datetime, timezone

import pdfplumber
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def _table_to_text(table: list[list]) -> str:
    rows = []
    for row in table:
        cells = ["" if c is None else str(c).strip() for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def load_text_pdf(pdf_path: str) -> list[Document]:
    doc_name = os.path.basename(pdf_path)
    upload_time = datetime.now(timezone.utc).isoformat()
    documents: list[Document] = []

    # 1. Page text (PyMuPDF is 0-indexed, so +1 for human-friendly pages)
    for page in PyMuPDFLoader(pdf_path).load():
        if not page.page_content.strip():
            continue
        documents.append(Document(
            page_content=page.page_content,
            metadata={
                "doc_name": doc_name,
                "doc_type": "text_pdf",
                "page_number": page.metadata.get("page", 0) + 1,
                "content_type": "text",
                "upload_time": upload_time,
            },
        ))

    # 2. Tables (pdfplumber)
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, table in enumerate(page.extract_tables()):
                text = _table_to_text(table)
                if not text.strip():
                    continue
                documents.append(Document(
                    page_content=f"[Table on page {page_number}]\n{text}",
                    metadata={
                        "doc_name": doc_name,
                        "doc_type": "text_pdf",
                        "page_number": page_number,
                        "content_type": "table",
                        "table_index": table_index,
                        "upload_time": upload_time,
                    },
                ))

    return documents
