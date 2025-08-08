# config.py
import os
from pathlib import Path

# --- Storage ---
FIG_STORE_PATH = os.getenv("FIG_STORE_PATH", "data/figures")  # local dev path
FIG_PUBLIC_BASE_URL = os.getenv("FIG_PUBLIC_BASE_URL", "")    # e.g., https://cdn.example.com/kelpgpt/figures

# --- Input PDFs ---
PDF_DIR = os.getenv("PDF_DIR", "data/papers")

# --- Chroma ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
FIGURES_COLLECTION = os.getenv("FIGURES_COLLECTION", "figures_v1")

# --- Figures extraction ---
USE_PDFFIGURES2 = os.getenv("USE_PDFFIGURES2", "false").lower() == "true"
MAX_FIGS_PER_PDF = int(os.getenv("MAX_FIGS_PER_PDF", "100"))

# --- Embeddings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" or "local"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")  # fast default

# --- Misc ---
THUMB_MAX_DIM = int(os.getenv("THUMB_MAX_DIM", "512"))  # thumbnail max side in px

# Ensure folders exist in dev mode
Path(FIG_STORE_PATH).mkdir(parents=True, exist_ok=True)
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
