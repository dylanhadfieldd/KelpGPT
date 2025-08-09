#!/usr/bin/env python3
"""
Indexes documents from data/papers into a persistent Chroma store.

- Uses LangChain (DirectoryLoader + text splitter + Chroma) if available.
- Falls back to a simple PDF/Text loader with direct chromadb client if LC isn't installed.
- Embeddings: OpenAI by default; local sentence-transformers if EMBEDDING_PROVIDER=local.

Env & paths come from config.py.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List

from config import (
    PAPERS_DIR, PAPERS_DIR_STR,
    CHROMA_PERSIST_DIR, CHROMA_PERSIST_DIR_STR,
    EMBEDDING_PROVIDER, OPENAI_API_KEY,
    OPENAI_EMBED_MODEL, LOCAL_EMBED_MODEL,
)

# ---- Embeddings (shared) -----------------------------------------------------
def _make_embeddings_fn():
    """
    Returns a callable: List[str] -> List[List[float]]
    Uses OpenAI or sentence-transformers based on EMBEDDING_PROVIDER.
    """
    if EMBEDDING_PROVIDER.lower() == "local":
        # Local: sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_EMBED_MODEL)

        def embed(batch: List[str]):
            return model.encode(batch, normalize_embeddings=True).tolist()

        return embed

    # OpenAI default
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(batch: List[str]):
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        return [d.embedding for d in resp.data]

    return embed


# ---- Attempt LangChain path first --------------------------------------------
def ingest_with_langchain() -> int:
    # Lazy imports so we can fall back if needed
    try:
        # loaders
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
        # chunking
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        # embeddings
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception:
            # legacy location in older LC versions
            from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # vector store
        from langchain_community.vectorstores import Chroma
    except Exception as e:
        raise RuntimeError(f"LangChain path unavailable: {e}")

    # Load docs (PDF, TXT, DOCX)
    loaders = [
        DirectoryLoader(PAPERS_DIR_STR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
        DirectoryLoader(PAPERS_DIR_STR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
    ]
    # docx is optional
    try:
        loaders.append(DirectoryLoader(PAPERS_DIR_STR, glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True))
    except Exception:
        pass

    docs = []
    for L in loaders:
        docs.extend(L.load())

    if not docs:
        print(f"No documents found under {PAPERS_DIR_STR}")
        return 0

    # Chunking (defaults fine; tweak if desired)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if EMBEDDING_PROVIDER.lower() == "local":
        embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)  # uses sentence-transformers
    else:
        # OpenAI embeddings
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR_STR,
    )
    # Ensure persisted
    vectordb.persist()

    print(f"Ingested {len(chunks)} chunks into {CHROMA_PERSIST_DIR_STR}")
    return len(chunks)


# ---- Minimal direct-Chroma fallback ------------------------------------------
def _load_pdf_text(pdf_path: Path) -> str:
    import pdfplumber
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
            if t.strip():
                out.append(t)
    return "\n\n".join(out)

def _split_text(text: str, size=1200, overlap=200) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end - overlap if end - overlap > i else end
    return chunks

def ingest_direct() -> int:
    import chromadb
    from uuid import uuid4

    embed = _make_embeddings_fn()
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR_STR)
    col = client.get_or_create_collection(name="papers_text")  # default; configurable later

    total = 0
    for pdf in sorted(Path(PAPERS_DIR).glob("**/*.pdf")):
        text = _load_pdf_text(pdf)
        chunks = _split_text(text)
        if not chunks:
            continue

        ids = [f"pdf::{pdf.name}::c{ix}::{uuid4().hex[:8]}" for ix, _ in enumerate(chunks)]
        embs = embed(chunks)
        metas = [{"source": pdf.as_posix(), "file_name": pdf.name, "chunk_index": ix, "page": None} for ix, _ in enumerate(chunks)]
        col.upsert(ids=ids, embeddings=embs, documents=chunks, metadatas=metas)
        total += len(chunks)

    print(f"Ingested {total} chunks into {CHROMA_PERSIST_DIR_STR}")
    return total


if __name__ == "__main__":
    try:
        n = ingest_with_langchain()
    except RuntimeError as _e:
        print(f"[ingest] Falling back to direct mode: {_e}")
        n = ingest_direct()
