#!/usr/bin/env python3
from __future__ import annotations
from typing import List
from pathlib import Path

# 1) Load .env early
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 2) Config after env
from config import (
    PAPERS_DIR, PAPERS_DIR_STR,
    CHROMA_PERSIST_DIR_STR,
    EMBEDDING_PROVIDER, OPENAI_API_KEY,
    OPENAI_EMBED_MODEL, LOCAL_EMBED_MODEL,
    TEXT_COLLECTION,
)

def _make_embeddings_fn():
    if EMBEDDING_PROVIDER.lower() == "local":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_EMBED_MODEL)
        def embed(batch: List[str]): return model.encode(batch, normalize_embeddings=True).tolist()
        return embed
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (add it to .env).")
    client = OpenAI(api_key=OPENAI_API_KEY)
    def embed(batch: List[str]):
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        return [d.embedding for d in resp.data]
    return embed

def ingest_with_langchain() -> int:
    try:
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception:
            from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
    except Exception as e:
        raise RuntimeError(f"LangChain path unavailable: {e}")

    loaders = [
        DirectoryLoader(PAPERS_DIR_STR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
        DirectoryLoader(PAPERS_DIR_STR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
    ]
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if EMBEDDING_PROVIDER.lower() == "local":
        embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)
    else:
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR_STR,
        collection_name=TEXT_COLLECTION,
    )
    vectordb.persist()
    print(f"Ingested {len(chunks)} chunks into {CHROMA_PERSIST_DIR_STR}/{TEXT_COLLECTION}")
    return len(chunks)

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
    chunks, i, n = [], 0, len(text)
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
    col = client.get_or_create_collection(name=TEXT_COLLECTION)

    total = 0
    for pdf in sorted(Path(PAPERS_DIR).glob("**/*.pdf")):
        text = _load_pdf_text(pdf)
        chunks = _split_text(text)
        if not chunks:
            continue
        ids = [f"pdf::{pdf.name}::c{ix}::{uuid4().hex[:8]}" for ix, _ in enumerate(chunks)]
        embs = embed(chunks)
        metas = [{"source": pdf.as_posix(), "file_name": pdf.name, "chunk_index": ix} for ix, _ in enumerate(chunks)]
        col.upsert(ids=ids, embeddings=embs, documents=chunks, metadatas=metas)
        total += len(chunks)

    print(f"Ingested {total} chunks into {CHROMA_PERSIST_DIR_STR}/{TEXT_COLLECTION}")
    return total

if __name__ == "__main__":
    try:
        n = ingest_with_langchain()
    except RuntimeError as e:
        print(f"[ingest] Falling back to direct mode: {e}")
        n = ingest_direct()
