"""Shared helper for vision pipelines.

Turns a file into base64-encoded image(s) ready to send to Gemini Vision:
  - PDFs are rasterized page-by-page (PyMuPDF) at the configured DPI.
  - Image files are encoded directly.

Used by both image_vision.py (description) and invoice.py (structured
extraction) so the encoding logic lives in one place.
"""
import base64
import os

import fitz  # PyMuPDF

from backend.config import settings


def images_from_file(path: str):
    """Yield (base64_string, mime_subtype) for each page/image in the file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        zoom = settings.OCR_DPI / 72
        with fitz.open(path) as pdf:
            for page in pdf:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                yield base64.b64encode(pix.tobytes("png")).decode("utf-8"), "png"
    else:
        mime = "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")
        with open(path, "rb") as f:
            yield base64.b64encode(f.read()).decode("utf-8"), mime