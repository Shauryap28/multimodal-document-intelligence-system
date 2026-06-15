# Multimodal Document Intelligence System

A RAG system for chatting with documents across many types and sources - text
PDFs, scanned pages, standalone images (typed or visual), invoices/forms, and
YouTube videos. Ask questions in natural language and get answers grounded in
whatever you fed in, with the sources each answer drew on.

For full design rationale, every parameter, technology choices with
alternatives, and a record of every problem and how it was resolved, see
[`DESIGN.md`](DESIGN.md).

---

## Architecture

A two-tier system. A thin Streamlit UI talks over HTTP to a FastAPI service that
owns all the RAG logic - the pipelines, embeddings, vector store, and LLM
access. Either tier can be run, swapped, or deployed independently; the UI
imports no backend code.

```
Streamlit UI  ──HTTP / JSON──►  FastAPI service
(api_client)                     ├─ ingest · query · documents routers
                                 ├─ owns embeddings · ChromaDB · LLM clients
                                 └─ six pipelines → chunk → BGE-small embeddings
                                    → ChromaDB → MMR retrieval → LCEL chain
                                    → Groq Llama 3.3 70B answer (+ sources)
```

---

## What it does

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

**Retrieval, conversation & management features:**
- **Source attribution** - each answer lists the documents/pages it was grounded in.
- **Conversation memory** - follow-up questions ("what is its total?") resolve against the chat history before retrieval.
- **Per-document query scoping** - ask within one document (metadata filter) or across all.
- **Per-document delete** and **clear** - manage the store without rebuilding it.
- **Duplicate-ingestion guard** - a document already indexed is not silently added twice.
- **Evaluation suite** - context-recall and answer-correctness metrics over a controlled corpus.

---

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit (HTTP client) |
| API | FastAPI + uvicorn |
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

---

## Running it

The backend and the UI run as two processes, both started from the project root:

```bash
# terminal 1 - the API
uvicorn backend.api.main:app --reload
#   interactive API docs at http://127.0.0.1:8000/docs

# terminal 2 - the UI
streamlit run frontend/streamlit_app.py
```

Point the UI at a non-local backend by setting `API_BASE_URL` (defaults to
`http://127.0.0.1:8000`). Terminal-only Q&A on a text PDF is available with
`python main.py` (expects `data/samples/sample.pdf`), which uses the RAG core
directly without the API.

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
  api/
    main.py            FastAPI app + lifespan resource warm-up
    deps.py            cached resource singletons
    schemas.py         Pydantic request/response models
    routers/           query.py  documents.py  ingest.py
frontend/
  streamlit_app.py     thin UI client
  api_client.py        HTTP wrapper around the API
data/  samples/  uploads/(gitignored)  chroma_db/(gitignored)
main.py  evaluate.py  requirements.txt  README.md  DESIGN.md  .env(gitignored)
```

---

## Known limitations

Documented honestly; several are deliberate scope decisions. Full analysis in
[`DESIGN.md`](DESIGN.md).

- **Aggregation/counting over a whole document** is unreliable - RAG retrieves
  relevant passages, not the full document (measured: aggregation answer
  correctness 0/2 with retrieval recall 100%). The proper fix is table-QA /
  text-to-SQL, deliberately out of scope.
- **Handwriting / complex layout** route to Gemini Vision, not EasyOCR.
- **Conversation memory is single-session** - follow-ups work within a session,
  but chat history resets on reload; there is no long-term memory across sessions.
- **Source attribution is document/page level**, not inline per-claim `[1][2]` markers.
- **The API has no auth, rate limiting, or pagination** - it is built for local,
  single-user use.
- **YouTube transcripts** can be blocked from cloud IPs (works locally; needs a
  residential proxy on cloud VMs).
- **Vision will not assert the identity of a depicted person/character** - by
  design; it reads visible text instead.

---

## Design highlights

For full rationale, technology choices, and the problem-resolution log, see
[`DESIGN.md`](DESIGN.md).

- **Two-tier decoupling** - the UI is a pure HTTP client; the FastAPI service is the single owner of the RAG logic and the ChromaDB, so clients and backend evolve independently.
- **Tiered OCR/Vision strategy** - EasyOCR for typed scans (free, local, unlimited); Gemini Vision for handwriting, structured extraction, and visual content. Right tool per input shape, not "which is better."
- **LLM factory** - `get_llm("text")` → Groq, `get_llm("vision")` → Gemini; the rest of the code is provider-agnostic.
- **Structured vs unstructured output** - invoices/forms get a Pydantic schema (queryable fields, with an `additional_details` catch-all so nothing is silently dropped); arbitrary images get a natural-language description.
- **Per-document scoping via pre-filtering** - the document filter is applied inside the vector search (Chroma `where`), so you always get k results from the right document.
- **Citations through chain composition** - the chain returns the answer *and* the retrieved documents (`RunnableParallel` + `RunnablePassthrough.assign`), so sources ride alongside the answer.
- **Conversation memory as query reformulation** - a history-aware step rewrites a follow-up into a standalone question before retrieval (`condense | qa`), fixing the retrieval bottleneck rather than stuffing history into the answer prompt.
- **ChromaDB over FAISS** - native metadata storage and filtering, persistence, and HNSW indexing without a separate server.
- **Document-as-universal-contract** - every pipeline emits the same `Document` shape, so new input types don't touch the downstream core.
- **Measured, not assumed** - limitations like multi-document interference (since fixed via scoping) and aggregation failure were found by controlled evaluation and documented.

---

## Future work

- **Containerization** - Docker + docker-compose to run the API and UI together.
- **Auto doc-type router** - heuristic detection instead of the manual selector.
- **Reranker** - a cross-encoder second stage, to be added only if evaluation shows it helps.
- **Multilingual** - BGE-M3 embeddings and multilingual OCR.
- **API hardening** - auth, rate limiting, pagination, versioning, and a test suite.