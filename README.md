# Multimodal Document Intelligence System

A RAG system for chatting with documents across multiple types — text PDFs,
scanned pages, invoices and forms, image-based content. Built as a portfolio
project for placements (ML / SDE roles).

> **Status:** Phase 1 complete — text PDFs (with table extraction), persistent
> vector store, LCEL chain, Streamlit chat UI. Future phases below.

---

## What works today (Phase 1)

- Upload a text PDF and chat with it in the browser
- Page text via PyMuPDF, table rows extracted separately via pdfplumber
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

# 4. Gemini key (free, no card) — get one at https://aistudio.google.com/apikey
echo GOOGLE_API_KEY=your_key_here > .env

# 5. Run the UI
streamlit run frontend/streamlit_app.py
```

For a terminal-only Q&A loop instead of the UI: `python main.py`
(expects a PDF at `data/samples/sample.pdf`).

---

## Project layout

```
backend/
  config/settings.py                  central config
  rag/
    embeddings.py                     BGE-small factory
    chunking.py                       recursive splitter + chunk_index
    vectorstore.py                    persistent Chroma, HNSW cosine
    retriever.py                      k=4 retriever
    chain.py                          LCEL RAG chain
  services/pipelines/
    text_pdf.py                       PyMuPDF text + pdfplumber tables
frontend/
  streamlit_app.py                    UI
data/
  samples/                            source PDFs
  uploads/                            UI-uploaded files (gitignored)
  chroma_db/                          persistent vectors (gitignored)
main.py                               CLI Q&A runner
requirements.txt
```

---

## Roadmap

- [x] **Phase 1** — Text PDFs (text + tables), persistent RAG, Streamlit UI
- [ ] **Phase 2** — EasyOCR pipeline for scanned documents
- [ ] **Phase 3** — Gemini Vision for invoices, forms, and embedded images
- [ ] **Phase 4** — Auto document router with user override
- [ ] **Phase 5** — FastAPI backend separation
- [ ] **Phase 6** — Per-session conversation memory (history-aware retrieval)
- [ ] **Phase 7** — Docker + docker-compose
- [ ] **Phase 8** — Citations / source attribution, evaluation suite, UI polish

---

## Design notes

**Why ChromaDB over FAISS?** Native metadata storage and filtering, per-collection
isolation, and HNSW indexing without separate setup. FAISS has no persistence or
metadata filtering out of the box.

**Why BGE-small (local) over OpenAI embeddings?** Free, runs on CPU, no API key,
no rate limit on the embedding layer. Strong English performance at this scale.

**Why Gemini 2.5 Flash?** Multimodal in a single model (the same LLM that
answers text questions in Phase 1 will read invoice images in Phase 3), and a
genuinely free tier for development.

**Why pdfplumber alongside PyMuPDF?** PyMuPDF is fast and accurate for prose
but mangles tables; pdfplumber walks tables row-by-row. They're complementary.

**Why LCEL instead of `ConversationalRetrievalChain`?** LCEL is the current
LangChain pattern; the legacy retrieval chains are deprecated. LCEL composes
cleanly — memory (Phase 6), reranking, and citations (Phase 8) plug in as
extra steps without rewriting.