"""Image vision pipeline (Gemini Vision).

For VISUAL content - charts, diagrams, infographics, handwriting, photographs,
mixed-content pages - where OCR alone is not enough and we want the model to
*understand* the image, not just read characters off it.

Now accepts PDFs as well as image files: a handwritten or complex-layout PDF
(which EasyOCR handles poorly) is rasterized page-by-page and each page is
described separately. This completes the "complex scan -> Gemini Vision" path.

Output is UNSTRUCTURED natural language (a description), NOT structured fields,
because an arbitrary image has no fixed schema. Contrast with invoice.py.
"""
import os
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from backend.rag.llm import get_llm
from backend.services.pipelines._vision_io import images_from_file


_DESCRIBE_PROMPT = (
    "You are creating a detailed text description of an image for a search index. "
    "Completeness matters more than brevity. Follow these rules:\n"
    "1. Transcribe ALL text visible in the image VERBATIM - titles, labels, "
    "captions, headings, names, author credits, and small print.\n"
    "2. If the image contains multiple labeled sections, diagrams, panels, or "
    "items, list EVERY ONE by its exact label and describe each individually. "
    "Do not summarize them as a group.\n"
    "3. For charts or graphs: state the chart type, the axes, and the data "
    "values or trends shown.\n"
    "4. For diagrams or flowcharts: describe each component and how they connect.\n"
    "5. Describe the overall layout, colors, and any other notable visual elements.\n"
    "Report what is visible and what the text says. You may state what a labeled "
    "diagram represents based on its label."
)


def load_image_vision(file_path: str) -> list[Document]:
    doc_name = os.path.basename(file_path)
    upload_time = datetime.now(timezone.utc).isoformat()
    llm = get_llm("vision")   # Gemini 2.5 Flash, larger output budget

    documents: list[Document] = []
    for page_number, (b64, mime) in enumerate(images_from_file(file_path), start=1):
        message = HumanMessage(content=[
            {"type": "text", "text": _DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": f"data:image/{mime};base64,{b64}"},
        ])
        response = llm.invoke([message])
        description = (response.content or "").strip()
        if not description:
            continue
        documents.append(Document(
            page_content=description,
            metadata={
                "doc_name": doc_name,
                "doc_type": "image_vision",
                "page_number": page_number,
                "content_type": "description",
                "upload_time": upload_time,
            },
        ))
    return documents