from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]       # repo root
APP_DIR = ROOT / "app"
STATIC_DIR = APP_DIR / "static"
DATA_DIR = ROOT / "data"
AVATARS_DIR = DATA_DIR / "avatars"
DB_DIR = ROOT / "db"
CHROMA_DIR = DATA_DIR / "chroma_db"

def ensure_dirs():
    AVATARS_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

