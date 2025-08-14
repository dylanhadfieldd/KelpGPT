# ingest.py
# -----------------------------------------------------------------------------
# Ingest PDFs into a persistent ChromaDB collection with OpenAI embeddings.
# - Reads .env for: OPENAI_API_KEY, CHROMA_PERSIST_DIR, TEXT_COLLECTION, EMBED_MODEL
# - Scans for PDFs under data/ and data/papers/ by default (recursive).
# - Extracts core PDF metadata (title, authors, year) and stores:
#     paper_title, authors, year, doc_id, page, chunk_index, source_filename
# - Uses stable chunk IDs: "{doc_id}::p{page:04d}::c{chunk:04d}" and UP SERTs.
# - Windows & Linux safe (absolute CHROMA path relative to this file).
# - CLI:
#     python ingest.py                       # scan default folders
#     python ingest.py --path path\to\*.pdf  # glob(s) or folder(s)
#     python ingest.py --clean doc_id_slug   # delete existing doc_id first
# -----------------------------------------------------------------------------

import os
import io
import re
import sys
import glob
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

# --- .env support (no Streamlit dependency here) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).parent.resolve()

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)

OPENAI_API_KEY  = _get_env("OPENAI_API_KEY", "")
RAW_CHROMA_DIR  = _get_env("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION = _get_env("TEXT_COLLECTION", "papers_text")
EMBED_MODEL     = _get_env("EMBED_MODEL", "text-embedding-3-small")  # 1536-d default

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY missing in environment or .env", file=sys.stderr)
    sys.exit(1)

# Make persist dir absolute and consistent with app.py
CHROMA_PERSIST_DIR = str((ROOT / RAW_CHROMA_DIR).resolve())

# --- Chroma + Embeddings ---
import chromadb
from chromadb.utils import embedding_functions

# --- PDF parsing ---
from pypdf import PdfReader

# ------------------------- Helpers -------------------------

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _slugify(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-") or "doc"

def _file_checksum(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _extract_pdf_core_metadata(file_bytes: bytes) -> Dict[str, Any]:
    meta = {"title": None, "authors": None, "year": None}
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        info = reader.metadata or {}
        title = str(info.get("/Title", "") or info.get("Title", "") or "").strip() or None
        authors = str(info.get("/Author","") or info.get("Author","") or "").strip() or None
        y = str(info.get("/CreationDate","") or info.get("CreationDate","") or "")
        year = None
        m = re.search(r"(19|20)\d{2}", y)
        if m: year = m.group(0)
        meta.update({"title": title, "authors": authors, "year": year})
    except Exception:
        pass
    return meta

def _extract_pages_text(path: Path) -> List[str]:
    """Return list of per-page extracted text (strings)."""
    texts: List[str] = []
    with path.open("rb") as fh:
        reader = PdfReader(fh)
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
    return texts

def _chunk_text_words(text: str, chunk_words: int = 900, overlap_words: int = 150) -> List[str]:
    """Word-based chunking; match app's style. Returns list of chunk strings."""
    words = text.split()
    out: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        j = min(n, i + chunk_words)
        out.append(" ".join(words[i:j]))
        if j == n: break
        i = j - overlap_words if j - overlap_words > i else j
    return [c for c in (s.strip() for s in out) if c]

def _iter_pdf_chunks(path: Path) -> Iterable[Tuple[int, int, str]]:
    """Yield (page_index, chunk_index, chunk_text) for each chunk."""
    pages = _extract_pages_text(path)
    for pidx, ptxt in enumerate(pages, start=1):
        if not ptxt.strip():
            continue
        chunks = _chunk_text_words(ptxt, chunk_words=900, overlap_words=150)
        for cidx, ctext in enumerate(chunks, start=1):
            yield pidx, cidx, _norm_ws(ctext)

def _doc_id_for(path: Path, meta: Dict[str, Any], checksum: str) -> str:
    """
    Stable-ish doc_id:
    - Prefer PDF title; fall back to filename stem.
    - Do NOT include checksum in doc_id (so an updated file replaces via --clean).
    - Keep checksum separately in metadata to detect changes.
    """
    base = meta.get("title") or path.stem
    return _slugify(base)  # e.g., "kelp-nitrogen-cycling-2021"

# ------------------------- Core ingest -------------------------

def build_embedding_fn() -> Any:
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )

def open_collection(ef) -> Any:
    cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = cli.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"embedding_model": EMBED_MODEL},
        embedding_function=ef
    )
    return cli, col

def delete_doc(col, doc_id: str) -> None:
    try:
        col.delete(where={"doc_id": {"$eq": doc_id}})
        print(f"Deleted existing records for doc_id='{doc_id}'")
    except Exception as e:
        print(f"Warning: delete failed for doc_id='{doc_id}': {e}")

def upsert_chunks(col, doc_id: str, file_path: Path, file_checksum: str, meta_core: Dict[str, Any]) -> int:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    source_filename = file_path.name

    total = 0
    for page_idx, chunk_idx, chunk_text in _iter_pdf_chunks(file_path):
        cid = f"{doc_id}::p{page_idx:04d}::c{chunk_idx:04d}"
        ids.append(cid)
        docs.append(chunk_text)
        metas.append({
            "doc_id": doc_id,
            "source_filename": source_filename,
            "file_path": str(file_path),
            "file_checksum": file_checksum,
            "paper_title": meta_core.get("title"),
            "authors": meta_core.get("authors"),
            "year": meta_core.get("year"),
            "page": page_idx,
            "chunk_index": chunk_idx,
            "embedding_model": EMBED_MODEL,
        })
        total += 1

        # Batch every ~512 to keep memory low on Pi
        if len(ids) >= 512:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)

    return total

# ------------------------- CLI / Main -------------------------

def find_pdf_paths(sources: List[str]) -> List[Path]:
    paths: List[Path] = []
    if not sources:
        # default: recurse under these folders
        defaults = [ROOT / "data", ROOT / "data" / "papers"]
        for d in defaults:
            paths.extend(d.rglob("*.pdf"))
    else:
        for src in sources:
            p = Path(src)
            if p.is_dir():
                paths.extend(p.rglob("*.pdf"))
            else:
                # allow glob strings (Windows-friendly)
                for g in glob.glob(src, recursive=True):
                    gp = Path(g)
                    if gp.is_dir():
                        paths.extend(gp.rglob("*.pdf"))
                    elif gp.suffix.lower() == ".pdf":
                        paths.append(gp)
    # de-dup, keep order-ish
    seen, out = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out

def main():
    ap = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB with OpenAI embeddings.")
    ap.add_argument("--path", "-p", nargs="*", default=[], help="Folder(s), glob(s), or PDF file(s) to ingest")
    ap.add_argument("--clean", "-c", default="", help="If set, delete existing records for this doc_id before ingest")
    ap.add_argument("--model", "-m", default=EMBED_MODEL, help="Embedding model name (default: env EMBED_MODEL)")
    args = ap.parse_args()

    global EMBED_MODEL
    EMBED_MODEL = args.model

    ef = build_embedding_fn()
    cli, col = open_collection(ef)

    pdfs = find_pdf_paths(args.path)

    if not pdfs:
        print("No PDFs found. Put files under ./data/ or ./data/papers/ or pass --path.", file=sys.stderr)
        return

    print(f"Indexing {len(pdfs)} PDF(s) into collection '{TEXT_COLLECTION}' at {CHROMA_PERSIST_DIR}")
    for path in pdfs:
        try:
            file_bytes = path.read_bytes()
            meta_core = _extract_pdf_core_metadata(file_bytes)
            checksum = _file_checksum(path)
            doc_id = _doc_id_for(path, meta_core, checksum)

            if args.clean and args.clean == doc_id:
                delete_doc(col, doc_id)

            count = upsert_chunks(col, doc_id, path, checksum, meta_core)
            print(f" - {path.name}: doc_id='{doc_id}', chunks={count}")

        except Exception as e:
            print(f"ERROR processing {path}: {e}", file=sys.stderr)

    # Print a tiny summary of the collection
    try:
        got = col.get(include=[])
        n = len((got or {}).get("ids") or [])
        print(f"Done. Collection now has ~{n} records.")
    except Exception:
        print("Done.")

if __name__ == "__main__":
    main()
