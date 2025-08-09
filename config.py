# config.py
import os
from pathlib import Path

# === Base data dirs (matches architecture) ===
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", DATA_DIR / "papers"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", DATA_DIR / "figures"))
THUMBS_DIR = Path(os.getenv("THUMBS_DIR", FIGURES_DIR / "_thumbs"))

# Public URL base for serving figures (leave empty to use local paths)
FIG_PUBLIC_BASE_URL = os.getenv("FIG_PUBLIC_BASE_URL", "").rstrip("/")

# === Chroma (unified name) ===
# Prefer CHROMA_PERSIST_DIR; fall back to your old CHROMA_PATH for compatibility
CHROMA_PERSIST_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR", os.getenv("CHROMA_PATH", "chroma_db"))
)

# Collections
TEXT_COLLECTION = os.getenv("TEXT_COLLECTION", "papers_text")
FIGURES_COLLECTION = os.getenv("FIGURES_COLLECTION", "figures_v1")

# === Figures extraction ===
USE_PDFFIGURES2 = os.getenv("USE_PDFFIGURES2", "false").lower() == "true"
MAX_FIGS_PER_PDF = int(os.getenv("MAX_FIGS_PER_PDF", "100"))

# === Embeddings ===
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" | "local"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

# === Thumbnails ===
THUMB_MAX_DIM = int(os.getenv("THUMB_MAX_DIM", "512"))  # px

# === Ensure folders exist (dev/local) ===
for p in (DATA_DIR, PAPERS_DIR, FIGURES_DIR, THUMBS_DIR, CHROMA_PERSIST_DIR):
    p.mkdir(parents=True, exist_ok=True)

# === Convenience: POSIX strings if some libs want str paths ===
DATA_DIR_STR = DATA_DIR.as_posix()
PAPERS_DIR_STR = PAPERS_DIR.as_posix()
FIGURES_DIR_STR = FIGURES_DIR.as_posix()
THUMBS_DIR_STR = THUMBS_DIR.as_posix()
CHROMA_PERSIST_DIR_STR = CHROMA_PERSIST_DIR.as_posix()
