# scripts/ingest.py
# -----------------------------------------------------------------------------
# Ingest PDFs into a persistent ChromaDB collection with OpenAI embeddings.
# - Reads .env for: OPENAI_API_KEY, CHROMA_PERSIST_DIR, TEXT_COLLECTION, EMBED_MODEL
# - Scans for PDFs under <repo>/data/ and <repo>/data/papers/ by default (recursive).
# - Stores per-chunk metadata: paper_title, authors, year, doc_id, page, chunk_index,
#   source_filename, embedding_model.
# - Windows & Linux safe. Paths resolve relative to the REPO ROOT (one level up).
# - CLI:
#     python scripts/ingest.py                   # scan default folders
#     python scripts/ingest.py --path data\*.pdf # specific glob(s) or folder(s)
#     python scripts/ingest.py --clean doc_id    # delete existing doc before ingest
#     python scripts/ingest.py --model text-embedding-3-large
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

# --- Resolve repo root even if this lives in scripts/ ---
FILE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = FILE_DIR.parent if FILE_DIR.name.lower() == "scripts" else FILE_DIR

# --- .env support (no Streamlit dependency here) ---
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")  # load from repo root if present
except Exception:
    pass

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)

OPENAI_API_KEY  = _get_env("OPENAI_API_KEY", "")
RAW_CHROMA_DIR  = _get_env("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION = _get_env("TEXT_COLLECTION", "papers_text")
DEFAULT_MODEL   = _get_env("EMBED_MODEL", "text-embedding-3-small")  # 1536-d default

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY missing in environment or .env", file=sys.stderr)
    sys.exit(1)

# Absolute persist dir (relative to repo root)
CHROMA_PERSIST_DIR = str((REPO_ROOT / RAW_CHROMA_DIR).resolve())

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

def _doc_id_for(path: Path, meta: Dict[str, Any]) -> str:
    base = meta.get("title") or path.stem
    return _slugify(base)  # e.g., "kelp-nitrogen-cycling-2021"

# ------------------------- Core ingest -------------------------

def build_embedding_fn(embed_model: str):
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=embed_model,
    )

def open_collection(ef, embed_model: str):
    cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = cli.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"embedding_model": embed_model},
        embedding_function=ef
    )
    return cli, col

def delete_doc(col, doc_id: str) -> None:
    try:
        col.delete(where={"doc_id": {"$eq": doc_id}})
        print(f"Deleted existing records for doc_id='{doc_id}'")
    except Exception as e:
        print(f"Warning: delete failed for doc_id='{doc_id}': {e}")

def upsert_chunks(col, doc_id: str, file_path: Path, file_checksum: str,
                  meta_core: Dict[str, Any], embed_model: str) -> int:
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
            "embedding_model": embed_model,
        })
        total += 1

        # Batch write to keep memory low
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
        # default: recurse under these folders (repo root)
        defaults = [REPO_ROOT / "data", REPO_ROOT / "data" / "papers"]
        for d in defaults:
            if d.exists():
                paths.extend(d.rglob("*.pdf"))
    else:
        for src in sources:
            p = Path(src)
            if p.is_dir():
                paths.extend(p.rglob("*.pdf"))
            else:
                for g in glob.glob(src, recursive=True):
                    gp = Path(g)
                    if gp.is_dir():
                        paths.extend(gp.rglob("*.pdf"))
                    elif gp.suffix.lower() == ".pdf":
                        paths.append(gp)
    # de-dup
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
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Embedding model name (default: env EMBED_MODEL)")
    args = ap.parse_args()

    embed_model = args.model

    ef = build_embedding_fn(embed_model)
    cli, col = open_collection(ef, embed_model)

    pdfs = find_pdf_paths(args.path)
    if not pdfs:
        print(f"No PDFs found. Put files under '{REPO_ROOT / 'data'}' or pass --path.", file=sys.stderr)
        return

    print(f"Indexing {len(pdfs)} PDF(s) into '{TEXT_COLLECTION}' at {CHROMA_PERSIST_DIR}")
    for path in pdfs:
        try:
            file_bytes = path.read_bytes()
            meta_core = _extract_pdf_core_metadata(file_bytes)
            checksum = _file_checksum(path)
            doc_id = _doc_id_for(path, meta_core)

            if args.clean and args.clean == doc_id:
                delete_doc(col, doc_id)

            count = upsert_chunks(col, doc_id, path, checksum, meta_core, embed_model)
            print(f" - {path.name}: doc_id='{doc_id}', chunks={count}")

        except Exception as e:
            print(f"ERROR processing {path}: {e}", file=sys.stderr)

    # tiny summary
    try:
        got = col.get(include=[])
        n = len((got or {}).get("ids") or [])
        print(f"Done. Collection now has ~{n} records.")
    except Exception:
        print("Done.")

if __name__ == "__main__":
    main()
