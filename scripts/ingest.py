# scripts/ingest.py
# -----------------------------------------------------------------------------
# Robust PDF -> ChromaDB ingest with small embedding batches & retry logic.
# - Tolerant to mildly corrupted PDFs (pypdf strict=False; skip unreadable pages).
# - Small, configurable embedding batches (default 64) to avoid huge HTTP posts.
# - Custom OpenAI embedder with retries, timeouts, and a .name() method
#   so Chroma's embedding-function validator is happy.
# - Metadata per chunk matches app.py: paper_title, authors, year, doc_id,
#   page, chunk_index, source_filename, embedding_model.
# -----------------------------------------------------------------------------

import os
import io
import re
import sys
import glob
import time
import math
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

# ---------- Resolve repo root even if this lives in scripts/ ----------
FILE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = FILE_DIR.parent if FILE_DIR.name.lower() == "scripts" else FILE_DIR

# ---------- .env support ----------
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)

OPENAI_API_KEY  = _get_env("OPENAI_API_KEY", "")
RAW_CHROMA_DIR  = _get_env("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION = _get_env("TEXT_COLLECTION", "papers_text")
DEFAULT_MODEL   = _get_env("EMBED_MODEL", "text-embedding-3-small")  # 1536-d

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY missing in environment or .env", file=sys.stderr)
    sys.exit(1)

# Absolute persist dir (relative to repo root)
CHROMA_PERSIST_DIR = str((REPO_ROOT / RAW_CHROMA_DIR).resolve())

# ---------- Chroma ----------
import chromadb

# ---------- OpenAI (custom embedder) ----------
from openai import OpenAI
from httpx import HTTPError

class OpenAIEmbedder:
    """Chroma-compatible embedding function with retries and timeouts."""
    def __init__(self, api_key: str, model: str, timeout: float = 45.0, max_retries: int = 5, backoff: float = 1.5):
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.backoff = backoff

    # Chroma calls this to check EF identity; returning a stable string avoids conflicts
    def name(self) -> str:
        return f"openai:{self.model}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        # One call per upsert() batch
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, input=input)
                return [d.embedding for d in resp.data]
            except (HTTPError, Exception):
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= self.backoff

# ---------- PDF parsing ----------
from pypdf import PdfReader
from pypdf.errors import PdfReadError

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
        # strict=False to be lenient with slightly malformed metadata
        reader = PdfReader(io.BytesIO(file_bytes), strict=False)
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
    """Return per-page text; tolerant to some corrupt PDFs."""
    texts: List[str] = []
    try:
        reader = PdfReader(path.open("rb"), strict=False)
    except PdfReadError as e:
        print(f"!! Skipping file (cannot open): {path.name} — {e}", file=sys.stderr)
        return texts
    except Exception as e:
        print(f"!! Skipping file (open error): {path.name} — {e}", file=sys.stderr)
        return texts

    for i, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception as e:
            print(f"!! Skipping page {i} in {path.name}: {e}", file=sys.stderr)
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
    if not pages:
        return
    for pidx, ptxt in enumerate(pages, start=1):
        if not ptxt.strip():
            continue
        chunks = _chunk_text_words(ptxt, chunk_words=900, overlap_words=150)
        for cidx, ctext in enumerate(chunks, start=1):
            yield pidx, cidx, _norm_ws(ctext)

def _doc_id_for(path: Path, meta: Dict[str, Any]) -> str:
    base = meta.get("title") or path.stem
    return _slugify(base)

# ------------------------- Core ingest -------------------------

def open_collection(embedder, embed_model: str):
    cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = cli.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"embedding_model": embed_model},  # reflect the actual model
        embedding_function=embedder,
    )
    return cli, col

def upsert_chunks(col, doc_id: str, file_path: Path, file_checksum: str,
                  meta_core: Dict[str, Any], embed_model: str, batch_size: int = 64) -> int:
    all_records: List[Tuple[str, str, Dict[str, Any]]] = []

    source_filename = file_path.name
    for page_idx, chunk_idx, chunk_text in _iter_pdf_chunks(file_path):
        cid = f"{doc_id}::p{page_idx:04d}::c{chunk_idx:04d}"
        meta = {
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
        }
        all_records.append((cid, chunk_text, meta))

    total = len(all_records)
    if total == 0:
        print(f"   (no readable text) {file_path.name}")
        return 0

    # Batched upserts so each embeddings call is small
    batches = math.ceil(total / batch_size)
    for bi in range(batches):
        start = bi * batch_size
        end = min(total, start + batch_size)
        batch = all_records[start:end]
        ids = [r[0] for r in batch]
        docs = [r[1] for r in batch]
        metas = [r[2] for r in batch]

        col.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"   • upserted batch {bi+1}/{batches} ({end-start} chunks)")

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
    ap.add_argument("--batch", "-b", type=int, default=64, help="Embedding/Upsert batch size (default 64)")
    args = ap.parse_args()

    embed_model = args.model
    batch_size = max(8, min(256, args.batch))  # sane bounds

    embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY, model=embed_model, timeout=45.0, max_retries=5, backoff=1.7)
    cli, col = open_collection(embedder, embed_model)

    pdfs = find_pdf_paths(args.path)
    if not pdfs:
        print(f"No PDFs found. Put files under '{REPO_ROOT / 'data'}' or pass --path.", file=sys.stderr)
        return

    print(f"Indexing {len(pdfs)} PDF(s) into '{TEXT_COLLECTION}' at {CHROMA_PERSIST_DIR}")
    for path in pdfs:
        try:
            file_bytes = path.read_bytes()
            meta_core = _extract_pdf_core_metadata(file_bytes)
            doc_id = _doc_id_for(path, meta_core)

            if args.clean and args.clean == doc_id:
                try:
                    col.delete(where={"doc_id": {"$eq": doc_id}})
                    print(f"Deleted existing records for doc_id='{doc_id}'")
                except Exception as e:
                    print(f"Warning: delete failed for doc_id='{doc_id}': {e}")

            checksum = _file_checksum(path)
            count = upsert_chunks(col, doc_id, path, checksum, meta_core, embed_model, batch_size=batch_size)
            print(f" - {path.name}: doc_id='{doc_id}', chunks={count}")

        except PdfReadError as e:
            print(f"!! Skipping file (PdfReadError): {path.name} — {e}", file=sys.stderr)
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
