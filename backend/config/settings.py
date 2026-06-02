"""Central configuration - the single source of truth."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Models ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemini-2.5-flash"

# --- Chunking ---
# Splitter uses LangChain default separators ["\n\n", "\n", " ", ""],
# which is correct for paragraph-heavy prose. See chunking.py for the
# code-heavy upgrade path.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Vector store ---
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "documents"
DISTANCE_METRIC = "cosine"   # matches the normalized BGE embeddings

# --- Retrieval ---
TOP_K = 4

# MMR (Maximal Marginal Relevance) retrieval params.
# MMR balances "relevance to the query" against "diversity from already-picked
# chunks", which prevents top-k from being four near-duplicate paragraphs.
MMR_FETCH_K = 20      # candidates fetched before diversification
MMR_LAMBDA = 0.5      # 1.0 = pure relevance, 0.0 = pure diversity; 0.5 is balanced

# --- Generation ---
MAX_OUTPUT_TOKENS = 512   # cap output during testing to save free-tier tokens

# --- OCR (scanned PDFs, Phase 2) ---
OCR_LANGUAGES = ["en"]      # EasyOCR language codes; add "hi", "fr" etc. as needed
OCR_DPI = 200               # page rasterization DPI; 200 balances quality vs speed
OCR_MIN_CONFIDENCE = 0.3    # drop OCR detections below this confidence (0.0-1.0)