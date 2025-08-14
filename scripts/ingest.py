# ingest.py
# ---------------------------------------------
# Ingest PDFs into a local Chroma DB with rich metadata
# - Stores human-readable paper_title and authors for better citations
# - Stable doc_id per PDF; safe to re-run (upsert)
# - Uses modern Chroma client (PersistentClient)
# ---------------------------------------------

import os
import re
import glob
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from pypdf import PdfReader

# Local dev: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions

# ----------- Settings / ENV -----------
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION    = os.getenv("TEXT_COLLECTION", "papers_text")
PAPERS_DIR         = Path(os.getenv("PAPERS_DIR", Path("data") / "papers"))
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # match app.py (1536-dim)

assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"

# ----------- Helpers -----------

def read_pdf(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Return raw text and raw PDF metadata dict."""
    reader = PdfReader(str(path))
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)

    md = {}
    try:
        info = reader.metadata or {}
        for k, v in (info or {}).items():
            key = str(k).strip("/").strip()
            md[key] = str(v) if v is not None else ""
    except Exception:
        pass
    return text, md


def guess_title_and_authors(raw_text: str, pdf_meta: Dict[str, Any], file_name: str = "") -> Tuple[str, str]:
    """
    Heuristic title/author extraction:
    1) Prefer PDF metadata fields when present.
    2) Else, infer title from top of page 1 (before 'Abstract', first ~60 lines).
    3) Authors: metadata; else a short names-like line near the title.
    4) Fallback: filename stem as title.
    """
    meta_title = (pdf_meta.get("Title") or pdf_meta.get("title") or "").strip()
    meta_auth  = (pdf_meta.get("Author") or pdf_meta.get("author") or "").strip()

    title = meta_title
    authors = meta_auth

    head = (raw_text or "")[:4000]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]

    if not title:
        cut = []
        for ln in lines[:60]:
            low = ln.lower()
            if re.match(r"^\s*abstract[:\s]*$", low) or low.startswith("abstract"):
                break
            cut.append(ln)
        candidates = [ln for ln in cut if 8 <= len(ln) <= 180 and not ln.endswith(".")]
        if candidates:
            title = candidates[0]

    if not title:
        title = Path(file_name).stem.replace("_", " ").replace("-", " ").strip() or "Untitled Paper"

    if not authors:
        cand = ""
        try_idx = min(lines.index(title) + 1 if title in lines else 0, len(lines)-1)
        scan = lines[try_idx: try_idx + 12]
        for ln in scan:
            low = ln.lower()
            if (ln.count(",") >= 1 or " and " in low) and len(ln) < 200:
                if not re.search(r"@|university|institute|department|laboratory", low):
                    cand = ln
                    break
        authors = cand.strip()

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

    return [b for b in blocks if b]


def stable_doc_id(file_path: Path) -> str:
    """Repeatable ID for file path + size + mtime."""
    stat = file_path.stat()
    sig = f"{str(file_path.resolve())}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def ensure_collection(client: "chromadb.PersistentClient", name: str):
    """Create or load collection and enforce embedding model consistency."""
    # Always use the same embedding function as app.py
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )
    try:
        coll = client.get_collection(name=name, embedding_function=ef)
    except Exception:
        coll = None

    if coll is not None:
        meta = coll.metadata or {}
        used_model = meta.get("embedding_model", "")
        if used_model and used_model != EMBED_MODEL:
            raise RuntimeError(
                f"Embedding model mismatch.\n"
                f"Persisted: {used_model}\nNew: {EMBED_MODEL}\n"
                f"To fix: delete/rename '{CHROMA_PERSIST_DIR}' or set EMBED_MODEL consistently."
            )
        return coll

    coll = client.get_or_create_collection(
        name=name,
        metadata={"embedding_model": EMBED_MODEL},
        embedding_function=ef,
    )
    return coll


def _gather_input_pdfs(argv: List[str]) -> List[Path]:
    """If argv contains files/globs, use those; else list PAPERS_DIR/*.pdf."""
    if argv:
        paths: List[Path] = []
        for pat in argv:
            for s in glob.glob(pat):
                p = Path(s)
                if p.is_file() and p.suffix.lower() == ".pdf":
                    paths.append(p)
        return sorted(set(paths))
    else:
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(Path(PAPERS_DIR).glob("*.pdf"))


def main():
    pdf_paths = _gather_input_pdfs(sys.argv[1:])
    print(f"Indexing {len(pdf_paths)} PDF(s) from {PAPERS_DIR} into collection '{TEXT_COLLECTION}'")

    # NEW Chroma client (no legacy Settings)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    coll = ensure_collection(client, TEXT_COLLECTION)

    added = 0
    for p in pdf_paths:
        p = Path(p)
        raw, meta = read_pdf(p)
        if not raw.strip():
            print(f" - Skipping (empty text): {p.name}")
            continue

        title, authors = guess_title_and_authors(raw, meta, p.name)
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

        coll.upsert(ids=ids, documents=docs, metadatas=metadatas)
        print(f" - Ingested: {p.name}  | title: {title}  | chunks: {len(chunks)}")
        added += len(chunks)

    print(f"Done. Total chunks upserted: {added}")
    print(f"Chroma DB at: {Path(CHROMA_PERSIST_DIR).resolve()}")


if __name__ == "__main__":
    main()
