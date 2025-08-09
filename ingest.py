# ingest_cli.py
import os, glob, hashlib
from typing import List
from pypdf import PdfReader


# load .env if present
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

def _extract_text(path: str) -> str:
    reader = PdfReader(path)
    out = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t: out.append(t)
    return "\n\n".join(out)

def _chunk(text: str, size=1400, overlap=200) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + size, len(words))
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks

def main(paths: List[str]):
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    coll = client.get_or_create_collection(name=TEXT_COLLECTION, embedding_function=ef)

    added = 0
    for p in paths:
        text = _extract_text(p)
        if not text.strip():
            print(f"Skip (no text): {p}")
            continue
        chunks = _chunk(text)
        with open(p, "rb") as fh:
            digest = hashlib.sha1(fh.read()).hexdigest()
        ids = [f"{digest}-{i}" for i in range(len(chunks))]
        metas = [{"filename": os.path.basename(p), "chunk": i} for i in range(len(chunks))]
        coll.upsert(documents=chunks, ids=ids, metadatas=metas)
        print(f"Ingested {len(chunks)} chunks from {p}")
        added += len(chunks)
    print(f"Done. Total chunks added: {added}")

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="PDF files or glob patterns")
    args = ap.parse_args()

    # expand globs on Windows too
    files = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))
    if not files:
        raise SystemExit("No matching PDFs found.")
    main(files)
