"""Image OCR pipeline (for direct .jpg / .png / .webp / .bmp uploads).

When a user uploads a photo or screenshot of typed text - a book page,
a sign, a printed paragraph - there is no need to wrap it in a PDF.
We OCR the image directly with the same EasyOCR engine used for scanned PDFs.

This is the Phase 2 pipeline minus the PDF rasterization step. It reuses
the lazy reader and OCR helper from scanned.py so the EasyOCR model is
loaded only once across both pipelines.

Visual content like charts, diagrams, infographics, and handwriting belong
to the Phase 3 image-vision pipeline (Gemini Vision), not here.
"""
import os
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from langchain_core.documents import Document

from backend.config import settings
from backend.services.pipelines.scanned import _ocr_image


def load_image(image_path: str) -> list[Document]:
    doc_name = os.path.basename(image_path)
    upload_time = datetime.now(timezone.utc).isoformat()

    # PIL handles jpg, png, webp, bmp, tiff transparently.
    # Convert to RGB to drop alpha and force 3 channels for EasyOCR.
    pil_image = Image.open(image_path).convert("RGB")
    image_array = np.array(pil_image)

    text = _ocr_image(image_array, settings.OCR_MIN_CONFIDENCE)
    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={
            "doc_name": doc_name,
            "doc_type": "image_text",
            "page_number": 1,         # a single image is one "page"
            "content_type": "text",
            "upload_time": upload_time,
        },
    )]