#!/usr/bin/env python3
"""
KelpGPT Ingest (matches app.py embedding model)

- Scans data/papers for PDFs by default (no args needed)
- Creates/updates per-PDF sidecar JSON (via metadata.py)
- Extracts text, chunks, embeds with OpenAI, stores in Chroma
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from pypdf import PdfReader
import chromadb

# === Config ===
try:
    import config
    DATA_DIR = Path(getattr(config, "DATA_DIR", "data"))
    PAPERS_DIR = Path(getattr(config, "PAPERS_DIR", DATA_DIR / "papers"))
    CHROMA_PERSIST_DIR = Path(getattr(config, "CHROMA_PERSIST_DIR", "chroma_db"))
    TEXT_COLLECTION = getattr(config, "TEXT_COLLECTION", "papers_text")
except Exception:
    DATA_DIR = Path("data")
    PAPERS_DIR = DATA_DIR / "papers"
    CHROMA_PERSIST_DIR = Path("chroma_db")
    TEXT_COLLECTION = "papers_text"

# === Metadata helper ===
try:
    from metadata import load_or_init, save_sidecar, ensure_reference_cache, sha256
except ImportError:
    print("ERROR: metadata.py not found. Please add it to your project.")
    sys.exit(1)

# === OpenAI ===
from openai import OpenAI
_openai_client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"  # matches app.py
EMBED_DIM = 1536  # text-embedding-3-small output dimension

# === Helpers ===
def ensure_dirs():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

def read_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text.strip())
    return pages

def simple_chunk(text: str, chunk_size: int = 1200, overlap: int = 120) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks

def chunk_pdf_pages(pages: List[str],
                    chunk_size: int = 1200,
                    overlap: int = 120) -> List[Tuple[int, int, str]]:
    out = []
    for p, text in enumerate(pages, start=1):
        if not text:
            continue
        parts = simple_chunk(text, chunk_size, overlap)
        for part in parts:
            out.append((p, p, part))
    return out

def embed_texts(batch: List[str]) -> List[List[float]]:
    resp = _openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
    return [d.embedding for d in resp.data]

def get_collection(rebuild: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    if rebuild:
        try:
            client.delete_collection(TEXT_COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(name=TEXT_COLLECTION)

def pdf_already_indexed(collection, doc_id: str, doc_hash: str) -> bool:
    try:
        res = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        metas = res.get("metadatas", []) or []
        return any(m.get("doc_hash") == doc_hash for m in metas)
    except Exception:
        return False

def chunk_id(doc_id: str, page_start: int, page_end: int, idx: int) -> str:
    return f"{doc_id}::p{page_start}-{page_end}::c{idx}"

# === Main ingest logic ===
def ingest_pdf(collection, pdf_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    meta = load_or_init(str(pdf_path))
    meta = ensure_reference_cache(meta)
    meta["pdf_path"] = str(pdf_path)
    doc_hash = meta.get("hash") or sha256(str(pdf_path))
    meta["hash"] = doc_hash
    doc_id = meta["doc_id"]

    if pdf_already_indexed(collection, doc_id, doc_hash):
        return {"doc_id": doc_id, "status": "skip", "reason": "already indexed"}

    pages = read_pdf_pages(pdf_path)
    if not any(pages):
        return {"doc_id": doc_id, "status": "warn", "reason": "no extractable text"}

    chunks = chunk_pdf_pages(pages)
    ids, docs, metas = [], [], []
    for idx, (p0, p1, text) in enumerate(chunks):
        ids.append(chunk_id(doc_id, p0, p1, idx))
        docs.append(text)
        metas.append({
            "doc_id": doc_id,
            "pdf_path": str(pdf_path),
            "page_start": p0,
            "page_end": p1,
            "doc_hash": doc_hash,
            "apa": meta.get("reference_style_cache", {}).get("apa"),
        })

    if dry_run:
        return {"doc_id": doc_id, "status": "dry-run", "chunks": len(docs)}

    # Embed in batches
    BATCH = 64
    for i in range(0, len(docs), BATCH):
        vecs = embed_texts(docs[i:i+BATCH])
        collection.add(
            ids=ids[i:i+BATCH],
            documents=docs[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
            embeddings=vecs
        )

    save_sidecar(str(pdf_path), meta)
    return {"doc_id": doc_id, "status": "ok", "chunks": len(docs)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Delete collection and reindex")
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

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skip")
    warn = sum(1 for r in results if r["status"] == "warn")
    err = sum(1 for r in results if r["status"] == "error")

    print(f"\nDone. ok={ok}, skipped={skipped}, warn={warn}, error={err}")
    for r in results:
        if r["status"] in ("warn", "error"):
            print(f" - {r.get('doc_id')}: {r.get('status')} -> {r.get('reason') or r.get('error')}")

if __name__ == "__main__":
    main()
