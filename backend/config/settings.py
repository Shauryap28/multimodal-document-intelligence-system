"""Central configuration - the single source of truth for Phase 1."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Models ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemini-2.5-flash"

# --- Generation ---
MAX_OUTPUT_TOKENS = 512   # cap output during testing to save free-tier tokens

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Vector store ---
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "documents"
DISTANCE_METRIC = "cosine"   # matches the normalized BGE embeddings

# --- Retrieval ---
TOP_K = 4
