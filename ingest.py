# ingest.py
# ---------------------------------------------
# Ingest PDFs into a local Chroma DB with rich metadata
# - Stores human-readable paper_title and authors for better citations
# - Uses a stable doc_id per PDF so you can re-ingest safely
# - Ensures the same embedding model is used across runs
# ---------------------------------------------

import os
import glob
import uuid
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

from pypdf import PdfReader

# Local dev: load .env if present (Windows-safe)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ----------- Settings / ENV -----------
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION    = os.getenv("TEXT_COLLECTION", "papers_text")
DATA_DIR           = Path(os.getenv("PAPERS_DIR", Path("data") / "papers"))
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-large")  # pin this

assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"

# ----------- Helpers -----------

def read_pdf(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Return raw text and raw PDF metadata dict."""
    reader = PdfReader(str(path))
    # Concatenate pages
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)

    # pypdf metadata
    md = {}
    try:
        info = reader.metadata or {}
        # Convert to regular dict with strings
        for k, v in (info or {}).items():
            key = str(k).strip("/")
            md[key] = str(v) if v is not None else ""
    except Exception:
        pass

    return text, md


def guess_title_and_authors(raw_text: str, pdf_meta: Dict[str, Any]) -> Tuple[str, str]:
    """Heuristic:
    1) Use PDF metadata Title/Author when available.
    2) Else, use the first non-empty line of the first ~1500 chars as title.
    3) Authors: try metadata; else look for a line with multiple commas near the top.
    """
    title = (pdf_meta.get("Title") or pdf_meta.get("title") or "").strip()
    authors = (pdf_meta.get("Author") or pdf_meta.get("author") or "").strip()

    head = (raw_text or "")[:2000]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]

    if not title and lines:
        # Often first line is the paper title
        title = lines[0]
        # If it's suspiciously short or shouty, try the second line
        if len(title) < 5 and len(lines) > 1:
            title = lines[1]

    if not authors:
        # crude guess: line with several commas (names separated)
        cand = ""
        for ln in lines[:15]:
            if ("," in ln and len(ln) < 200) or (" and " in ln.lower()):
                cand = ln
                break
        authors = cand

    # Clean fallbacks
    if not title:
        title = "Untitled Paper"
    if not authors:
        authors = ""

    return title.strip(), authors.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Simple, robust chunker for RAG."""
    text = text.replace("\r", "")
    tokens = text.split("\n")
    blocks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for ln in tokens:
        ln_len = len(ln)
        if cur_len + ln_len + 1 > chunk_size:
            blocks.append("\n".join(cur).strip())
            # overlap by characters (approx via lines)
            if overlap > 0:
                joined = "\n".join(cur).strip()
                tail = joined[-overlap:]
                cur = [tail] if tail else []
                cur_len = len(tail)
            else:
                cur, cur_len = [], 0
        cur.append(ln)
        cur_len += ln_len + 1

    if cur:
        blocks.append("\n".join(cur).strip())

    # filter empties
    return [b for b in blocks if b]


def stable_doc_id(file_path: Path) -> str:
    """Create a repeatable ID for a file path + size + mtime."""
    stat = file_path.stat()
    sig = f"{str(file_path.resolve())}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def ensure_collection(client: Client, name: str):
    """Create or load collection and enforce embedding model consistency."""
    try:
        coll = client.get_collection(name)
    except Exception:
        coll = None

    if coll is not None:
        meta = coll.metadata or {}
        used_model = meta.get("embedding_model", "")
        if used_model and used_model != EMBED_MODEL:
            raise RuntimeError(
                f"Embedding model mismatch.\n"
                f"Persisted: {used_model}\nNew: {EMBED_MODEL}\n"
                f"To fix: re-ingest with the same model, or delete the collection at {CHROMA_PERSIST_DIR}."
            )
        return coll

    # Create new collection
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL
    )
    coll = client.create_collection(
        name=name,
        metadata={"embedding_model": EMBED_MODEL},
        embedding_function=ef
    )
    return coll


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(glob.glob(str(DATA_DIR / "*.pdf")))
    print(f"Indexing {len(pdf_paths)} PDF(s) from {DATA_DIR} into collection '{TEXT_COLLECTION}'")

    client = Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PERSIST_DIR,
    ))

    # Important: attach embedding function at collection creation time
    coll = ensure_collection(client, TEXT_COLLECTION)

    added = 0
    for p in pdf_paths:
        p = Path(p)
        raw, meta = read_pdf(p)
        if not raw.strip():
            print(f" - Skipping (empty text): {p.name}")
            continue

        title, authors = guess_title_and_authors(raw, meta)
        chunks = chunk_text(raw)

        did = stable_doc_id(p)
        ids = []
        docs = []
        metadatas = []
        for i, ch in enumerate(chunks):
            cid = f"{did}-{i:05d}"
            ids.append(cid)
            docs.append(ch)
            metadatas.append({
                "doc_id": did,
                "paper_title": title,
                "authors": authors,
                "file_name": p.name,
                "abs_path": str(p.resolve()),
                "chunk_index": i
            })

        # Upsert (safe to re-run)
        coll.upsert(ids=ids, documents=docs, metadatas=metadatas)
        print(f" - Ingested: {p.name}  | title: {title}  | chunks: {len(chunks)}")
        added += len(chunks)

    client.persist()
    print(f"Done. Total chunks upserted: {added}")
    print(f"Chroma DB at: {Path(CHROMA_PERSIST_DIR).resolve()}")


if __name__ == "__main__":
    main()
