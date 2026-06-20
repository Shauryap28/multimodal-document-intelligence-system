"""Central configuration - the single source of truth."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Models ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"    # text RAG (Phases 1, 2, YouTube)
GEMINI_VISION_MODEL = "gemini-2.5-flash"        # vision tasks (Phase 3)

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Vector store ---
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "documents"
DISTANCE_METRIC = "cosine"   # matches the normalized BGE embeddings

# --- Retrieval ---
TOP_K = 4
MMR_FETCH_K = 20
MMR_LAMBDA = 0.5

# --- Conversation memory (Phase 7) ---
# How many recent turns (one turn = a user question + its assistant answer) to
# feed the question-reformulation step. Follow-ups usually reference the last
# 1-2 turns; 5 gives headroom while keeping the rewrite prompt short and cheap.
HISTORY_WINDOW = 5

# --- Hybrid search (semantic + BM25) ---
# When True, retrieval fuses dense (Chroma) with BM25 keyword search via
# reciprocal rank fusion, to recover exact-token matches (SKUs, emails, codes)
# that dense embeddings alone miss. Measured to lift exact-token recall.
HYBRID_SEARCH = True
HYBRID_WEIGHTS = [0.5, 0.5]   # [dense, sparse] - kept EQUAL for balance
# RRF constant. Lower sharpens rank-1 dominance (a confident top hit from either
# channel counts for more) without biasing toward either channel. Default is 60;
# 30 helps a BM25-only rank-1 chunk surface. Sweep lower (20, 15) if needed.
HYBRID_RRF_C = 30

# --- Generation ---
# Per-query answers stay short to save free-tier tokens.
MAX_OUTPUT_TOKENS = 512
# Image descriptions are a ONE-TIME ingestion cost and should be complete,
# so they get a much larger budget than per-query answers.
VISION_MAX_OUTPUT_TOKENS = 2048

# --- OCR (scanned PDFs / image-text, Phase 2) ---
OCR_LANGUAGES = ["en"]
OCR_DPI = 200
OCR_MIN_CONFIDENCE = 0.3