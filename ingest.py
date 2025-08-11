# ingest.py
# Ingest PDFs into a local Chroma collection using OpenAI embeddings (1536-dim).
# Usage:
#   python ingest.py data/papers/*.pdf
#   python ingest.py "data/papers/**/*.pdf"

import os, glob, hashlib, io
from typing import List, Dict, Any
from pathlib import Path

from pypdf import PdfReader

# Load .env if present (ignored in git)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION    = os.getenv("TEXT_COLLECTION", "papers_text")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in env/.env")

def _extract_text_from_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    out = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            out.append(t)
    return "\n\n".join(out)

def _parse_pdf_year(raw_date: str | None) -> str | None:
    import re
    if not raw_date:
        return None
    m = re.search(r"(19|20)\d{2}", raw_date)
    return m.group(0) if m else None

def _extract_core_metadata_from_pdf_file(path: Path) -> Dict[str, Any]:
    try:
        reader = PdfReader(str(path))
        md = reader.metadata or {}
        title = (getattr(md, "title", None) or md.get("/Title") or "").strip() or None
        authors = (getattr(md, "author", None) or md.get("/Author") or "").strip() or None
        year = _parse_pdf_year(getattr(md, "creation_date", None) or md.get("/CreationDate"))
        return {"title": title, "authors": authors, "year": year}
    except Exception:
        return {"title": None, "authors": None, "year": None}

def _chunk(text: str, size=1400, overlap=200) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + size, len(words))
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks

def ingest(paths: List[Path]) -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",  # 1536-dim
    )
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    coll = client.get_or_create_collection(name=TEXT_COLLECTION, embedding_function=ef)

    total = 0
    for p in paths:
        if not p.exists() or not p.is_file() or p.suffix.lower() != ".pdf":
            print(f"Skip (not a PDF file): {p}")
            continue

        try:
            text = _extract_text_from_pdf_file(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue

        if not text.strip():
            print(f"Skip (no text): {p}")
            continue

        chunks = _chunk(text)
        # Content-addressable id from file bytes
        digest = hashlib.sha1(p.read_bytes()).hexdigest()
        ids = [f"{digest}-{i}" for i in range(len(chunks))]

        core = _extract_core_metadata_from_pdf_file(p)
        metadatas = []
        for i in range(len(chunks)):
            metadatas.append({
                "source_filename": p.name,
                "chunk": i,
                # APA-ish fields (optional but nice for display)
                "authors": core.get("authors"),
                "year": core.get("year"),
                "title": core.get("title"),
            })

        coll.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"Ingested {len(chunks)} chunks from {p}")
        total += len(chunks)

    print(f"Done. Total chunks added: {total}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="PDF files or glob patterns")
    args = ap.parse_args()

    files: list[Path] = []
    for pat in args.inputs:
        files.extend([Path(x) for x in glob.glob(pat, recursive=True)])
    if not files:
        raise SystemExit("No matching PDFs found.")
    ingest(files)
