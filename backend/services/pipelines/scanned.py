"""Scanned-PDF pipeline.

Pipeline:
    page (PDF)  ->  PyMuPDF rasterize to image  ->  EasyOCR -> text  ->  Document

A "scanned PDF" has no extractable text layer - the pages are images. So we:
  1. Render each page to a numpy image at a chosen DPI (PyMuPDF).
  2. Run EasyOCR on the image to recover the text.
  3. Wrap each page's text in a Document with the project's metadata schema.
The chunker handles splitting downstream, identical to the text_pdf pipeline.
"""
import os
from datetime import datetime, timezone

import fitz                 # PyMuPDF (imported as `fitz` historically)
import easyocr
import numpy as np
from langchain_core.documents import Document

from backend.config import settings


# Module-level reader: EasyOCR loads ~120 MB of model weights into memory.
# We initialize lazily on first use, then reuse across calls and across
# Streamlit reruns (Python caches modules in sys.modules, so this global
# is not re-initialized on every rerun).
_reader = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=False)
    return _reader


def _page_to_image(page: "fitz.Page", dpi: int) -> np.ndarray:
    """Render a PDF page to a numpy image array EasyOCR can consume."""
    # PyMuPDF renders at a `zoom` factor where 1.0 == 72 DPI.
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    # RGBA -> RGB if needed (EasyOCR expects 3-channel)
    if pix.n == 4:
        image = image[:, :, :3]
    return image


def _ocr_image(image: np.ndarray, min_confidence: float) -> str:
    """Run EasyOCR on one image; return joined text above the confidence floor."""
    reader = _get_reader()
    # readtext returns: [(bbox, text, confidence), ...]
    results = reader.readtext(image)
    lines = [text for _, text, conf in results if conf >= min_confidence]
    return "\n".join(lines)


def load_scanned_pdf(pdf_path: str) -> list[Document]:
    doc_name = os.path.basename(pdf_path)
    upload_time = datetime.now(timezone.utc).isoformat()
    documents: list[Document] = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            image = _page_to_image(page, settings.OCR_DPI)
            text = _ocr_image(image, settings.OCR_MIN_CONFIDENCE)
            if not text.strip():
                continue   # blank page or OCR found nothing readable
            documents.append(Document(
                page_content=text,
                metadata={
                    "doc_name": doc_name,
                    "doc_type": "scanned",
                    "page_number": page_number,
                    "content_type": "text",
                    "upload_time": upload_time,
                },
            ))

    return documents