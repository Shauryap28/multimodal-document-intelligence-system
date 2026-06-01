# Multimodal Document Intelligence System

A RAG system for chatting with documents across multiple types — text PDFs,
scanned pages, invoices and forms, image-based content. Built as a portfolio
project for placements (ML / SDE roles).

> **Status:** Phases 1 and 2 complete — text PDFs (with table extraction),
> scanned PDFs (via EasyOCR), persistent vector store, LCEL chain, manual
> doc-type router, Streamlit chat UI. See [Roadmap](#roadmap) for upcoming phases.

For deep design rationale on every tool and parameter, see [`DESIGN.md`](DESIGN.md).

---

## What works today

- Upload a text PDF or a scanned PDF and chat with it in the browser
- **Text PDFs**: page text via PyMuPDF, table rows extracted separately via pdfplumber
- **Scanned PDFs**: page rasterization via PyMuPDF, OCR via EasyOCR (English; Hindi available as a config toggle, multilingual retrieval listed under future upgrades)
- Manual doc-type router in the sidebar (Text PDF / Scanned); auto-detect comes in Phase 4
- Local BGE-small embeddings (no embedding-API cost)
- Persistent ChromaDB store with HNSW cosine index
- LCEL RAG chain (`ChatPromptTemplate` + retriever + LLM + `StrOutputParser`)
- Gemini 2.5 Flash for generation, output-token-capped for free-tier safety
- "Clear store" control to reset between documents

---

## Tech stack

| Layer            | Tool                                         |
|------------------|----------------------------------------------|
| UI               | Streamlit                                    |
| Orchestration    | LangChain (LCEL Runnables)                   |
| Text extraction  | PyMuPDF                                      |
| Table extraction | pdfplumber                                   |
| OCR              | EasyOCR (CRAFT + CRNN)                       |
| Embeddings       | BGE-small-en-v1.5 (sentence-transformers)    |
| Vector store     | ChromaDB (HNSW, cosine)                      |
| LLM              | Gemini 2.5 Flash                             |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<you>/multimodal-document-intelligence-system.git
cd multimodal-document-intelligence-system

# 2. Virtual env (Python 3.10+)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Gemini key (free, no card) - get one at https://aistudio.google.com/apikey
echo GOOGLE_API_KEY=your_key_here > .env

# 5. Run the UI
streamlit run frontend/streamlit_app.py
```

For a terminal-only Q&A loop on text PDFs: `python main.py`
(expects a PDF at `data/samples/sample.pdf`).

First-run notes: ~130 MB of BGE-small embedding weights and ~120 MB of EasyOCR
models download on first use and are cached locally; subsequent runs are offline
for those layers.

---

## Project layout

```
backend/
  config/settings.py                  central config (chunking, retrieval, OCR, LLM)
  rag/
    embeddings.py                     BGE-small factory
    chunking.py                       recursive splitter + chunk_index
    vectorstore.py                    persistent Chroma, HNSW cosine
    retriever.py                      k=4 retriever
    chain.py                          LCEL RAG chain
  services/pipelines/
    text_pdf.py                       PyMuPDF text + pdfplumber tables
    scanned.py                        PyMuPDF rasterize + EasyOCR
frontend/
  streamlit_app.py                    UI with doc-type router
data/
  samples/                            source PDFs
  uploads/                            UI-uploaded files (gitignored)
  chroma_db/                          persistent vectors (gitignored)
main.py                               CLI Q&A runner
requirements.txt
DESIGN.md                             detailed design rationale
```

---

## Roadmap

- [x] **Phase 1** — Text PDFs (text + tables), persistent RAG, Streamlit UI, LCEL chain
- [x] **Phase 2** — EasyOCR pipeline for scanned PDFs + manual doc-type router
- [ ] **Phase 3** — Gemini Vision for invoices, forms, embedded images, and complex scans
- [ ] **Phase 4** — Auto document router (heuristic detect + user override)
- [ ] **Phase 5** — FastAPI backend separation
- [ ] **Phase 6** — Per-session conversation memory (history-aware retrieval)
- [ ] **Phase 7** — Docker + docker-compose
- [ ] **Phase 8** — Citations / source attribution, evaluation suite, UI polish

---

## Future upgrades

These are deliberate "v2" choices documented for completeness, so the current
scope is honest about what is and isn't yet in the build.

- OCR: EasyOCR → Surya (better layout and reading-order detection)
- Embeddings: BGE-small-en → BGE-M3 or multilingual-e5 (multilingual retrieval)
- Retrieval: add BM25 hybrid search with RRF fusion
- Re-ranking: BGE-Reranker as a second-stage retriever
- Memory: ConversationBufferWindowMemory → ConversationSummaryBufferMemory
- Vector DB: ChromaDB → Qdrant (built-in hybrid search, production-grade)
- Local LLM stack: Gemini → Ollama (Llama 3.2) for fully-offline deployment

---

## Design highlights

For full rationale on every parameter and tool, see [`DESIGN.md`](DESIGN.md).
Quick summary of the more interesting decisions:

- **ChromaDB over FAISS** — native metadata storage and filtering, per-collection isolation, and HNSW indexing without separate setup. FAISS has no persistence or metadata filtering out of the box.
- **BGE-small (local) over OpenAI embeddings** — free, runs on CPU, no API key, no rate limit on the embedding layer. Strong English performance at this scale.
- **EasyOCR over Tesseract / PaddleOCR** — deep-learning-based and pure-pip-installable. Tesseract is rule-based and pre-deep-learning. PaddleOCR is more accurate but its framework (PaddlePaddle) creates Docker / CUDA friction not worth the gain at this scale.
- **Gemini 2.5 Flash over GPT-4o** — multimodal in a single model (the same LLM that answers text questions in Phase 1 will read invoice images in Phase 3), genuine free tier sufficient for development.
- **pdfplumber alongside PyMuPDF** — PyMuPDF is fast and accurate for prose but mangles tables. pdfplumber walks tables row-by-row. They are complementary, not alternatives.
- **LCEL instead of `ConversationalRetrievalChain`** — LCEL is the current LangChain pattern. The legacy retrieval chains are deprecated. LCEL composes cleanly so memory (Phase 6), reranking, and citations (Phase 8) plug in as extra pipe steps without rewriting.
- **Manual doc-type router (Phase 2)** — heuristic auto-detection lives in Phase 4. Phase 2 deliberately ships the manual version first so the routing and dispatch is wired and tested before the detection logic gets layered on top.