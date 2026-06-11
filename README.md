# Multimodal Document Intelligence System

A RAG system for chatting with documents across many types and sources - text
PDFs, scanned pages, standalone images (typed or visual), invoices/forms, and
YouTube videos. Ask questions in natural language and get answers grounded in
whatever you fed in.

> **Status:** Phase 7 complete. Six ingestion pipelines, two LLM providers
> (Groq for text, Gemini for vision), structured and unstructured extraction, a
> manual doc-type router, per-document query scoping, an evaluation suite,
> answers that cite their sources, conversation memory for follow-up questions,
> and a Streamlit chat UI. See [Roadmap](#roadmap).

For full design rationale, every parameter, technology choices with
alternatives, and a record of every problem and how it was resolved, see
[`DESIGN.md`](DESIGN.md).

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

**Retrieval & management features:**
- **Per-document query scoping** - ask within one document (metadata filter) or across all
- **Per-document delete** - remove one document without clearing the whole store
- **Duplicate-ingestion guard** - a document already indexed is not silently added twice

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
| Retrieval | MMR + optional per-document metadata filter |
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
GROQ_API_KEY=gsk_...           # https://console.groq.com/keys      (text RAG)
GOOGLE_API_KEY=...             # https://aistudio.google.com/apikey  (vision)
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
    embeddings.py  chunking.py  vectorstore.py    (+ list_documents, delete_document)
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
- [x] **Phase 4** - Per-document query scoping (metadata filter) + per-document delete + duplicate guard
- [x] **Phase 6** - Citations / source attribution (source-level)
- [x] **Phase 7** - Conversation memory (history-aware retrieval)
- [ ] **Phase 8** - FastAPI backend separation  ← next
- [ ] **Phase 9** - Docker + docker-compose
- [ ] **Future** - Auto doc-type router (heuristic detection); reranker (add only if measured to help); multilingual embeddings (BGE-M3)

Ordering note: the evaluation suite comes before the packaging phases (FastAPI,
Docker) on purpose - it's the measurement foundation that lets every later
change be judged on numbers, and packaging doesn't affect answer quality.

---

## Known limitations (current state)

Documented honestly; several are deliberate scope decisions. Full analysis in
[`DESIGN.md`](DESIGN.md).

- **Aggregation/counting over a whole document** is unreliable (RAG retrieves
  relevant passages, not the full document). *Measured in Phase 5 (aggregation
  answer correctness 0/2 with retrieval recall 100%); proper fix is table-QA /
  text-to-SQL, out of scope.*
- **Handwriting / complex layout** route to Gemini Vision, not EasyOCR.
- **Memory is single-session** - follow-up questions work within a session, but chat history resets on reload; there is no long-term memory across sessions.
- **Citations are source-level** (which documents/pages an answer drew on), not inline per-claim `[1][2]` markers - inline is a stretch goal.
- **YouTube transcripts** can be blocked from cloud IPs (works locally; needs a residential proxy on cloud VMs).
- **Vision will not assert the identity of a depicted person/character** - by design; it reads visible text instead.

> Multi-document interference (retrieval pulling unrelated chunks) was a measured
> limitation in earlier phases - **fixed in Phase 4** via per-document query scoping.

---

## Design highlights

For full rationale, technology choices, and the problem-resolution log, see
[`DESIGN.md`](DESIGN.md).

- **Tiered OCR/Vision strategy** - EasyOCR for typed scans (free, local, unlimited); Gemini Vision for handwriting, structured extraction, and visual content. Right tool per input shape, not "which is better."
- **LLM factory** - `get_llm("text")` → Groq, `get_llm("vision")` → Gemini; the rest of the code is provider-agnostic.
- **Structured vs unstructured output** - invoices/forms get a Pydantic schema (queryable fields); arbitrary images get a natural-language description (no schema).
- **Near-lossless structured extraction** - an `additional_details` catch-all field captures everything the fixed schema would otherwise drop.
- **Per-document scoping via pre-filtering** - the document filter is applied inside the vector search (Chroma `where`), not after, so you always get k results from the right document.
- **Answers cite their sources** - the chain returns the answer *and* the retrieved documents (via `RunnableParallel` + `RunnablePassthrough.assign`), so the UI shows which documents/pages each answer drew on.
- **Follow-up questions work** - a history-aware reformulation step rewrites "what is its total?" into a standalone question before retrieval, so references resolve; it reuses the citations chain unchanged (`condense | qa`) and only spends an extra call on follow-up turns.
- **ChromaDB over FAISS** - native metadata storage and filtering, persistence, and HNSW indexing without a separate server; FAISS would mean rebuilding all of that for speed we can't feel at this scale.
- **Document-as-universal-contract** - every pipeline emits the same `Document` shape, so new input types don't touch the downstream core.
- **Measured limitations** - multi-document interference and aggregation failures were found by controlled testing and documented (and the first was then fixed), not hidden.