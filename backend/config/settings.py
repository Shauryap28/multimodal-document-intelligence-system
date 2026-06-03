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

# MMR (Maximal Marginal Relevance) retrieval params.
MMR_FETCH_K = 20      # candidates fetched before diversification
MMR_LAMBDA = 0.5      # 1.0 = pure relevance, 0.0 = pure diversity

# --- Generation ---
MAX_OUTPUT_TOKENS = 512   # cap output during testing

# --- OCR (scanned PDFs, Phase 2) ---
OCR_LANGUAGES = ["en"]
OCR_DPI = 200
OCR_MIN_CONFIDENCE = 0.3