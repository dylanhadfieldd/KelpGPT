#!/usr/bin/env python3
"""
KelpGPT Ingest (minimal-disruption edition)

What this script does:
1) Scans your papers folder for PDFs
2) Creates/updates a per-PDF sidecar JSON (via metadata.py)
3) Extracts text (page by page), chunks it, and writes to ChromaDB
4) Skips re-embedding when the same PDF hash is already indexed

Requires:
- OPENAI_API_KEY in env (or Streamlit Secrets if you import it elsewhere)
- pip install: openai chromadb pypdf tqdm

Optional:
- config.py with DATA_DIR, PAPERS_DIR, CHROMA_PERSIST_DIR, TEXT_COLLECTION

Run:
    python ingest.py
    python ingest.py --rebuild            # wipe collection then reindex
    python ingest.py --pattern "*Ulva*.pdf"
"""

import os
import sys
import glob
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Optional config import (falls back if missing) ---
try:
    import config  # your existing file
    DATA_DIR = Path(getattr(config, "DATA_DIR", "data"))
    PAPERS_DIR = Path(getattr(config, "PAPERS_DIR", DATA_DIR / "papers"))
    CHROMA_PERSIST_DIR = Path(getattr(config, "CHROMA_PERSIST_DIR", "chroma_db"))
    TEXT_COLLECTION = getattr(config, "TEXT_COLLECTION", "papers_text")
except Exception:
    DATA_DIR = Path("data")
    PAPERS_DIR = DATA_DIR / "papers"
    CHROMA_PERSIST_DIR = Path("chroma_db")
    TEXT_COLLECTION = "papers_text"

# --- Minimal helper dependency (from our Option A) ---
try:
    from metadata import load_or_init, save_sidecar, ensure_reference_cache, sha256
except Exception as e:
    print("ERROR: metadata.py not found or import failed.\n"
          "Please add metadata.py (the small helper with sidecar JSON functions).")
    raise

# --- 3rd-party imports ---
from pypdf import PdfReader
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions

# OpenAI embeddings client (new SDK)
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception as e:
    print("ERROR: OpenAI client import failed. `pip install openai` and set OPENAI_API_KEY.")
    raise


# =========================
# Utilities
# =========================

def read_pdf_pages(pdf_path: Path) -> List[str]:
    """Return a list of page strings."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text.strip())
    return pages


def simple_chunk(text: str, chunk_size: int = 1200, overlap: int = 120) -> List[str]:
    """
    Simple char-based chunker to avoid extra deps.
    Keeps overlap to preserve context.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pdf_pages(pages: List[str],
                    chunk_size: int = 1200,
                    overlap: int = 120) -> List[Tuple[int, int, str]]:
    """
    Returns a list of (page_start, page_end, chunk_text).
    For a basic approach, we chunk each page independently (page_start == page_end).
    """
    out = []
    for p, text in enumerate(pages, start=1):
        if not text:
            continue
        parts = simple_chunk(text, chunk_size, overlap)
        for part in parts:
            out.append((p, p, part))
    return out


def embed_texts(batch: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    """
    Batches can be up to ~2048 items depending on token limits; keep batches small.
    """
    resp = _openai_client.embeddings.create(model=model, input=batch)
    return [d.embedding for d in resp.data]


def pdf_already_indexed(collection, doc_id: str, doc_hash: str) -> bool:
    """
    Check if we already indexed a document with this doc_id and hash.
    We store the doc_hash on each chunk; if any chunk matches and covers the doc_hash, assume done.
    """
    try:
        res = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        metas = res.get("metadatas", []) or []
        return any(m.get("doc_hash") == doc_hash for m in metas)
    except Exception:
        return False


def chunk_id(doc_id: str, page_start: int, page_end: int, idx: int) -> str:
    return f"{doc_id}::p{page_start}-{page_end}::c{idx}"


# =========================
# Main ingest
# =========================

def ensure_dirs():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def get_collection(rebuild: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    if rebuild:
        try:
            client.delete_collection(TEXT_COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
        # We’ll use manual OpenAI calls, so no built-in embedding function here.
    )
    return collection


def ingest_pdf(collection, pdf_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Process a single PDF into chunks + embeddings + Chroma entries."""
    # 1) Load or initialize sidecar metadata
    meta = load_or_init(str(pdf_path))
    meta = ensure_reference_cache(meta)
    meta["pdf_path"] = str(pdf_path)
    doc_hash = meta.get("hash") or sha256(str(pdf_path))
    meta["hash"] = doc_hash

    doc_id = meta["doc_id"]

    # 2) Skip if already indexed with same hash
    if pdf_already_indexed(collection, doc_id, doc_hash):
        return {"doc_id": doc_id, "status": "skip", "reason": "already indexed with same hash"}

    # 3) Extract text and chunk
    pages = read_pdf_pages(pdf_path)
    if not any(pages):
        return {"doc_id": doc_id, "status": "warn", "reason": "no extractable text"}

    chunks = chunk_pdf_pages(pages, chunk_size=1200, overlap=120)

    # 4) Embed and write to Chroma in small batches
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for idx, (p0, p1, text) in enumerate(chunks):
        ids.append(chunk_id(doc_id, p0, p1, idx))
        docs.append(text)
        metas.append({
            "doc_id": doc_id,
            "pdf_path": str(pdf_path),
            "page_start": p0,
            "page_end": p1,
            "doc_hash": doc_hash,
            # light hint for UI rendering/citation
            "apa": meta.get("reference_style_cache", {}).get("apa"),
        })

    if dry_run:
        return {"doc_id": doc_id, "status": "dry-run", "chunks": len(docs)}

    # Embed in batches of e.g., 64
    BATCH = 64
    for i in range(0, len(docs), BATCH):
        batch_ids = ids[i:i+BATCH]
        batch_docs = docs[i:i+BATCH]
        batch_meta = metas[i:i+BATCH]

        vectors = embed_texts(batch_docs, model="text-embedding-3-large")
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=vectors)

    # 5) Save (update) sidecar
    save_sidecar(str(pdf_path), meta)

    return {"doc_id": doc_id, "status": "ok", "chunks": len(docs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Delete the collection and reindex everything")
    parser.add_argument("--pattern", default="*.pdf", help="Glob pattern within PAPERS_DIR (default: *.pdf)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk, but don't write to Chroma")
    args = parser.parse_args()

    ensure_dirs()
    collection = get_collection(rebuild=args.rebuild)

    pdfs = sorted(PAPERS_DIR.glob(args.pattern))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR} matching {args.pattern}")
        sys.exit(0)

    print(f"Indexing {len(pdfs)} PDF(s) from {PAPERS_DIR} into collection '{TEXT_COLLECTION}'")
    results = []
    for pdf in tqdm(pdfs, desc="Ingesting"):
        try:
            out = ingest_pdf(collection, pdf, dry_run=args.dry_run)
        except Exception as e:
            out = {"doc_id": pdf.stem, "status": "error", "error": str(e)}
        results.append(out)

    # Summary
    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skip")
    warn = sum(1 for r in results if r.get("status") == "warn")
    err = sum(1 for r in results if r.get("status") == "error")

    print(f"\nDone. ok={ok}, skipped={skipped}, warn={warn}, error={err}")
    for r in results:
        if r.get("status") in ("warn", "error"):
            print(f" - {r.get('doc_id')}: {r.get('status')} -> {r.get('reason') or r.get('error')}")


if __name__ == "__main__":
    main()
