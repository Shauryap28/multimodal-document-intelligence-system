"""Invoice / form structured-extraction pipeline (Gemini Vision).

Extracts STRUCTURED fields from invoices, receipts, and forms using Gemini
Vision + with_structured_output() bound to a Pydantic schema.

LESSON FROM TESTING: a rigid schema is LOSSY - anything not in the schema is
dropped. Real invoices had emails, a discount, project details, and payment
info that the first schema discarded, causing "I don't know" answers and even
a wrong total (the missing discount made the math look inconsistent, so the
answer LLM recomputed and overrode the correct stored total).

FIX: this schema adds (a) the obviously-missing fields - discount, vendor/
customer contact - and (b) an `additional_details` CATCH-ALL that transcribes
everything else on the document. Core fields stay structured/queryable; nothing
gets dropped. Both go into the SUMMARY (page_content), because the answer chain
reads page_content, not metadata.

Accepts image files and PDFs (each page processed separately).
"""
import os
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from backend.rag.llm import get_llm
from backend.services.pipelines._vision_io import images_from_file


# --- Structured schema ---
class LineItem(BaseModel):
    description: str = Field(description="What the line item is")
    quantity: Optional[float] = Field(None, description="Quantity, if shown")
    unit_price: Optional[float] = Field(None, description="Price per unit, if shown")
    amount: Optional[float] = Field(None, description="Line total, if shown")


class Invoice(BaseModel):
    vendor: Optional[str] = Field(None, description="Seller / vendor / company name")
    vendor_contact: Optional[str] = Field(
        None, description="Vendor email, phone, website, and/or address - all of it"
    )
    customer: Optional[str] = Field(None, description="Buyer / customer name")
    customer_contact: Optional[str] = Field(
        None, description="Customer email, phone, and/or address - all of it"
    )
    invoice_number: Optional[str] = Field(None, description="Invoice or receipt number")
    invoice_date: Optional[str] = Field(None, description="Invoice date, as written")
    due_date: Optional[str] = Field(None, description="Payment due date, if present")
    currency: Optional[str] = Field(None, description="Currency symbol or code, e.g. $ or INR")
    subtotal: Optional[float] = Field(None, description="Subtotal before discount and tax")
    discount: Optional[float] = Field(None, description="Discount amount, if any (as a positive number)")
    tax: Optional[float] = Field(None, description="Total tax amount")
    total: Optional[float] = Field(None, description="Grand total amount due, exactly as printed")
    line_items: list[LineItem] = Field(
        default_factory=list, description="Itemized list of goods or services"
    )
    additional_details: Optional[str] = Field(
        None,
        description=(
            "Transcribe ANYTHING ELSE visible on the document that does not fit the "
            "fields above - payment terms, project details, dates, payment/bank "
            "instructions, reference/PO numbers, notes, totals in words, etc. "
            "Do not omit information."
        ),
    )


_EXTRACT_PROMPT = (
    "Extract structured data from this invoice/receipt/form image. "
    "Read every visible field carefully. If a field is not present, leave it null. "
    "Record the total exactly as printed - do not recompute it. "
    "Capture every line item. "
    "Put vendor and customer contact info (email, phone, address, website) in the "
    "contact fields. "
    "In 'additional_details', transcribe everything else on the document that does "
    "not fit the other fields. Do not omit information."
)


def _summarize(inv: Invoice) -> str:
    """Embeddable natural-language summary from the structured fields.

    Everything answerable MUST appear here, because the answer chain reads
    page_content, not metadata.
    """
    parts = []
    if inv.vendor:           parts.append(f"Vendor: {inv.vendor}.")
    if inv.vendor_contact:   parts.append(f"Vendor contact: {inv.vendor_contact}.")
    if inv.customer:         parts.append(f"Customer: {inv.customer}.")
    if inv.customer_contact: parts.append(f"Customer contact: {inv.customer_contact}.")
    if inv.invoice_number:   parts.append(f"Invoice number: {inv.invoice_number}.")
    if inv.invoice_date:     parts.append(f"Invoice date: {inv.invoice_date}.")
    if inv.due_date:         parts.append(f"Due date: {inv.due_date}.")
    cur = inv.currency or ""
    if inv.subtotal is not None: parts.append(f"Subtotal: {cur}{inv.subtotal}.")
    if inv.discount is not None: parts.append(f"Discount: {cur}{inv.discount}.")
    if inv.tax is not None:      parts.append(f"Tax: {cur}{inv.tax}.")
    if inv.total is not None:    parts.append(f"Total (as printed): {cur}{inv.total}.")
    if inv.line_items:
        parts.append("Line items:")
        for it in inv.line_items:
            bits = [it.description]
            if it.quantity is not None:   bits.append(f"qty {it.quantity}")
            if it.unit_price is not None: bits.append(f"unit {it.unit_price}")
            if it.amount is not None:     bits.append(f"amount {it.amount}")
            parts.append("  - " + ", ".join(bits))
    if inv.additional_details:
        parts.append(f"Additional details: {inv.additional_details}")
    return "\n".join(parts)


def _scalar_metadata(inv: Invoice, doc_name: str, page_number: int, upload_time: str) -> dict:
    """ChromaDB metadata must be scalar - skip None values and lists (line_items)."""
    meta = {
        "doc_name": doc_name,
        "doc_type": "invoice_form",
        "page_number": page_number,
        "content_type": "structured",
        "upload_time": upload_time,
    }
    for field in ("vendor", "customer", "invoice_number", "invoice_date",
                  "due_date", "currency", "subtotal", "discount", "tax", "total"):
        value = getattr(inv, field)
        if value is not None:
            meta[field] = value
    return meta


def load_invoice(file_path: str) -> list[Document]:
    doc_name = os.path.basename(file_path)
    upload_time = datetime.now(timezone.utc).isoformat()
    extractor = get_llm("vision").with_structured_output(Invoice)

    documents: list[Document] = []
    for page_number, (b64, mime) in enumerate(images_from_file(file_path), start=1):
        message = HumanMessage(content=[
            {"type": "text", "text": _EXTRACT_PROMPT},
            {"type": "image_url", "image_url": f"data:image/{mime};base64,{b64}"},
        ])
        invoice: Invoice = extractor.invoke([message])
        summary = _summarize(invoice)
        if not summary.strip():
            continue
        documents.append(Document(
            page_content=summary,
            metadata=_scalar_metadata(invoice, doc_name, page_number, upload_time),
        ))
    return documents