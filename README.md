# Multimodal Document Intelligence System

A RAG system for chatting with documents across many types and sources - text
PDFs, scanned pages, standalone images (typed or visual), invoices/forms, and
YouTube videos. Built as a portfolio project for placements (ML / SDE roles).

> **Status:** Phase 3 complete. Six ingestion pipelines, two LLM providers
> (Groq for text, Gemini for vision), structured and unstructured extraction,
> a manual doc-type router, and a Streamlit chat UI. See [Roadmap](#roadmap).

For full design rationale, every parameter, and a record of every problem we
hit and how we resolved it, see [`DESIGN.md`](DESIGN.md).

---

## What works today

| Input type | Pipeline | Engine |
|---|---|---|
| Text PDF | text + tables | PyMuPDF + pdfplumber |
| Scanned PDF | OCR (typed text) | EasyOCR |
| Image (text) | OCR on a photo/screenshot of text | EasyOCR |
| Visual / Handwritten | description of charts, diagrams, handwriting; accepts images **and** PDFs | Gemini Vision |
| Invoice / Form | structured field extraction; accepts images **and** PDFs | Gemini Vision + `with_structured_output` |
| YouTube | auto-generated transcript | youtube-transcript-api |

All six converge on the same core: chunk → BGE-small embeddings → ChromaDB →
MMR retrieval → LCEL chain → Groq Llama 3.3 70B answer.

---

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (LCEL) |
| Text extraction | PyMuPDF |
| Table extraction | pdfplumber |
| OCR | EasyOCR (CRAFT + CRNN) |
| Vision | Gemini 2.5 Flash |
| YouTube transcripts | youtube-transcript-api |
| Embeddings | BGE-small-en-v1.5 (local, 384-dim) |
| Vector store | ChromaDB (HNSW, cosine) |
| Retrieval | MMR |
| LLM (text RAG) | Groq Llama 3.3 70B |

---

## Setup

```bash
git clone https://github.com/<you>/multimodal-document-intelligence-system.git
cd multimodal-document-intelligence-system

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Two free API keys go in a `.env` file at the project root:
```
GROQ_API_KEY=gsk_...           # https://console.groq.com/keys   (text RAG)
GOOGLE_API_KEY=...             # https://aistudio.google.com/apikey (vision)
```

Run the UI:
```bash
streamlit run frontend/streamlit_app.py
```
Terminal-only Q&A on a text PDF: `python main.py` (expects `data/samples/sample.pdf`).

First run downloads ~130 MB (BGE-small) and ~120 MB (EasyOCR), cached locally
thereafter. Groq and Gemini are API-based (no local model).

---

## Project layout

```
backend/
  config/settings.py
  rag/
    embeddings.py  chunking.py  vectorstore.py
    retriever.py   chain.py     llm.py            (Groq for text, Gemini for vision)
  services/pipelines/
    text_pdf.py        PyMuPDF text + pdfplumber tables
    scanned.py         PyMuPDF rasterize + EasyOCR
    image_ocr.py       EasyOCR on a direct image upload
    image_vision.py    Gemini Vision description (images + PDFs)
    invoice.py         Gemini Vision structured extraction (images + PDFs)
    youtube.py         transcript fetch
    _vision_io.py      shared: file -> base64 image(s)  (DRY helper)
frontend/streamlit_app.py
data/  samples/  uploads/(gitignored)  chroma_db/(gitignored)
main.py  requirements.txt  README.md  DESIGN.md  .env(gitignored)
```

---

## Roadmap

- [x] **Phase 1** - Text PDFs (text + tables), persistent RAG, LCEL chain, UI
- [x] **Phase 2** - EasyOCR for scanned PDFs + manual doc-type router
- [x] **Phase 2.5** - Multi-LLM factory (Groq + Gemini), YouTube pipeline, MMR
- [x] **Phase 3** - Gemini Vision: image description, invoice structured extraction, PDF support for vision
- [ ] **Phase 4** - Per-document metadata filtering + auto doc-type router
- [ ] **Phase 5** - FastAPI backend separation
- [ ] **Phase 6** - Conversation memory (history-aware retrieval)
- [ ] **Phase 7** - Docker + docker-compose
- [ ] **Phase 8** - Citations, evaluation suite, UI polish

---

## Known limitations (current state)

Documented honestly; several are deliberate scope decisions. Full analysis in
[`DESIGN.md`](DESIGN.md).

- **Multi-document interference** - retrieval searches across all stored docs, so
  unrelated chunks can crowd out relevant ones. *Fix scheduled: Phase 4 per-document filtering.*
- **Aggregation/counting over a whole document** is unreliable (RAG retrieves
  relevant passages, not the full document). *Left as a documented limitation;
  proper fix is table-QA / text-to-SQL, out of scope.*
- **Handwriting / complex layout** route to Gemini Vision, not EasyOCR.
- **No conversation memory** yet (Phase 6); **no citations** yet (Phase 8).
- **YouTube transcripts** can be blocked from cloud IPs (works locally; needs a residential proxy on cloud VMs).
- **Vision will not assert the identity of a depicted person/character** - by design; it reads visible text instead.

---

## Design highlights

For full rationale and the problem-resolution log, see [`DESIGN.md`](DESIGN.md).

- **Tiered OCR/Vision strategy** - EasyOCR for typed scans (free, local, unlimited); Gemini Vision for handwriting, structured extraction, and visual content. Right tool per input shape, not "which is better."
- **LLM factory** - `get_llm("text")` → Groq, `get_llm("vision")` → Gemini; the rest of the code is provider-agnostic.
- **Structured vs unstructured output** - invoices/forms get a Pydantic schema (queryable fields); arbitrary images get a natural-language description (no schema).
- **Near-lossless structured extraction** - an `additional_details` catch-all field captures everything the fixed schema would otherwise drop.
- **Document-as-universal-contract** - every pipeline emits the same `Document` shape, so new input types don't touch the downstream core.
- **Measured limitations** - multi-document interference and aggregation failures were found by controlled testing and documented, not hidden.