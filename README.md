# Multimodal Document Intelligence System

A RAG system for chatting with documents across multiple types and sources —
text PDFs, scanned pages, YouTube videos, and (in coming phases) invoices,
forms, and image-based content. 

> **Status:** Phase 2.5 complete — text PDFs (with tables), scanned PDFs
> (EasyOCR), YouTube transcripts, persistent vector store, LCEL chain, MMR
> retrieval, multi-provider LLM (Groq + Gemini), manual doc-type router,
> Streamlit chat UI. See [Roadmap](#roadmap) for upcoming phases.

For deep design rationale on every tool and parameter, see [`DESIGN.md`](DESIGN.md).

---

## What works today

- Upload a **text PDF**, a **scanned PDF**, or paste a **YouTube URL** and chat about it in the browser
- **Text PDFs**: page text via PyMuPDF, tables extracted separately via pdfplumber
- **Scanned PDFs**: page rasterization via PyMuPDF + EasyOCR (English; Hindi available as a config toggle)
- **YouTube**: auto-generated transcripts via `youtube-transcript-api` (no API key, no quota)
- Manual doc-type router in the sidebar (auto-detect comes in Phase 4)
- Local BGE-small embeddings (no embedding-API cost)
- Persistent ChromaDB store with HNSW cosine index
- **MMR retrieval** — diversifies top-k results to avoid near-duplicates
- LCEL RAG chain (`ChatPromptTemplate` + retriever + LLM + `StrOutputParser`)
- **Multi-provider LLM**: Groq Llama 3.3 70B for text RAG (fast, free-tier-friendly), Gemini 2.5 Flash reserved for Phase 3 vision
- "Clear store" control to reset between documents

---

## Tech stack

| Layer                  | Tool                                          |
|------------------------|-----------------------------------------------|
| UI                     | Streamlit                                     |
| Orchestration          | LangChain (LCEL Runnables)                    |
| Text extraction        | PyMuPDF                                       |
| Table extraction       | pdfplumber                                    |
| OCR                    | EasyOCR (CRAFT + CRNN)                        |
| YouTube transcripts    | youtube-transcript-api                        |
| Embeddings             | BGE-small-en-v1.5 (sentence-transformers)     |
| Vector store           | ChromaDB (HNSW, cosine)                       |
| Retrieval              | MMR (Maximal Marginal Relevance)              |
| LLM (text RAG)         | Groq Llama 3.3 70B                            |
| LLM (vision, Phase 3)  | Gemini 2.5 Flash                              |

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

# 4. API keys (both free, no card)
#    Groq:   https://console.groq.com/keys
#    Gemini: https://aistudio.google.com/apikey
# Add both to .env at project root:
#   GROQ_API_KEY=gsk_...
#   GOOGLE_API_KEY=...

# 5. Run the UI
streamlit run frontend/streamlit_app.py
```

For a terminal-only Q&A loop on text PDFs: `python main.py`
(expects a PDF at `data/samples/sample.pdf`).

First-run notes: ~130 MB of BGE-small embedding weights and ~120 MB of EasyOCR
models download on first use and are cached locally; subsequent runs are
offline for those layers. Groq inference is API-based — no local model.

---

## Project layout

```
backend/
  config/settings.py                  central config
  rag/
    embeddings.py                     BGE-small factory
    chunking.py                       recursive splitter + chunk_index
    vectorstore.py                    persistent Chroma, HNSW cosine
    retriever.py                      MMR retriever
    chain.py                          LCEL RAG chain
    llm.py                            LLM factory (Groq for text, Gemini for vision)
  services/pipelines/
    text_pdf.py                       PyMuPDF text + pdfplumber tables
    scanned.py                        PyMuPDF rasterize + EasyOCR
    youtube.py                        YouTube transcript fetch
frontend/
  streamlit_app.py                    UI with doc-type router (file + URL inputs)
data/
  samples/                            source PDFs
  uploads/                            UI-uploaded files (gitignored)
  chroma_db/                          persistent vectors (gitignored)
main.py                               CLI Q&A runner (text PDFs)
requirements.txt
DESIGN.md                             detailed design rationale
```

---

## Roadmap

- [x] **Phase 1** — Text PDFs (text + tables), persistent RAG, LCEL chain, Streamlit UI
- [x] **Phase 2** — EasyOCR pipeline for scanned PDFs + manual doc-type router
- [x] **Phase 2.5** — Multi-provider LLM factory (Groq + Gemini), YouTube transcript pipeline, MMR retrieval
- [ ] **Phase 3** — Gemini Vision for invoices, forms, embedded images, and complex/handwritten scans
- [ ] **Phase 4** — Auto document router (heuristic detect + user override)
- [ ] **Phase 5** — FastAPI backend separation
- [ ] **Phase 6** — Per-session conversation memory (history-aware retrieval)
- [ ] **Phase 7** — Docker + docker-compose
- [ ] **Phase 8** — Citations / source attribution, evaluation suite, UI polish

---

## Future upgrades

- OCR: EasyOCR → Surya (better layout and reading-order detection)
- Embeddings: BGE-small-en → BGE-M3 or multilingual-e5 (full multilingual RAG)
- Retrieval: add BM25 hybrid + RRF fusion, optional BGE-Reranker second stage
- Memory: ConversationBufferWindowMemory → ConversationSummaryBufferMemory
- Vector DB: ChromaDB → Qdrant (production-grade, built-in hybrid)
- Local LLM stack: Groq → Ollama (Llama 3.x) for a fully-offline deployment
- Web page loader: a thin pipeline using `WebBaseLoader` for any URL

---

## Design highlights

For full rationale on every parameter and tool, see [`DESIGN.md`](DESIGN.md).
Quick summary of the most interview-worthy decisions:

- **Tiered OCR strategy** — EasyOCR for typed scans (free, local, unlimited); Gemini Vision (Phase 3) for handwriting, structured extraction, and visual content. Two tools because they're optimized for different shapes of input, not because one is "better."
- **LLM factory** — `get_llm("text")` returns Groq Llama 3.3 70B; `get_llm("vision")` returns Gemini 2.5 Flash. The rest of the codebase is provider-agnostic.
- **ChromaDB over FAISS** — native metadata storage and filtering, per-collection isolation, HNSW indexing without separate setup.
- **BGE-small (local) over OpenAI embeddings** — free, runs on CPU, no API key, no rate limit on the embedding layer.
- **MMR retrieval over plain cosine** — diversifies top-k to avoid four near-duplicate chunks landing in the LLM's context.
- **pdfplumber alongside PyMuPDF** — complementary, not alternative; tables run on different scaffolding from prose.
- **LCEL over `ConversationalRetrievalChain`** — current pattern; the legacy chain is deprecated. LCEL also composes cleanly for Phase 6 memory and Phase 8 citations.
- **YouTube via youtube-transcript-api** — free, no API key; demonstrates the Document-as-universal-contract architecture (totally new data source = one new file).
- **Manual doc-type router for now** — heuristic auto-detection lives in Phase 4. Shipping dispatch before detection means the routing is tested before the smarts go on top.