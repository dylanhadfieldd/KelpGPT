# services/rag.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional, Tuple
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from utils.paths import CHROMA_DIR

# ---- Embeddings client (OpenAI) ----
@dataclass
class EmbeddingClient:
    model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    api_key: str = os.getenv("OPENAI_API_KEY", "")

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Use Chroma's OpenAI helper so you don't reinvent clients
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key, model_name=self.model
        )
        return ef(texts)

# ---- Vector store wrapper (Chroma) ----
class VectorStore:
    def __init__(self,
                 collection: str = os.getenv("TEXT_COLLECTION", "papers_text"),
                 persist_dir: str = str(CHROMA_DIR)):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self,
               docs: List[Dict[str, Any]],
               embedder: Optional[EmbeddingClient] = None):
        """
        docs: [{id, text, metadata}]
        """
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]
        embeddings = None
        if embedder:
            embeddings = embedder.embed(texts)
        self.collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)

    def query(self,
              text: str,
              top_k: int = 5,
              embedder: Optional[EmbeddingClient] = None,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if embedder is None:
            # Let Chroma compute embeddings internally via server-side model if configured,
            # but we default to client-side for repeatability.
            raise ValueError("Provide an EmbeddingClient for client-side embeddings.")
        emb = embedder.embed([text])[0]
        return self.collection.query(query_embeddings=[emb], n_results=top_k, where=where or {})

# ---- High-level helpers for ingestion & search ----
def ingest_chunks(chunks: Iterable[Dict[str, Any]],
                  store: Optional[VectorStore] = None,
                  embedder: Optional[EmbeddingClient] = None):
    """
    chunks: iterable of {id, text, metadata}
    """
    store = store or VectorStore()
    embedder = embedder or EmbeddingClient()
    batch: List[Dict[str, Any]] = []
    for ch in chunks:
        batch.append(ch)
        if len(batch) >= 64:
            store.upsert(batch, embedder)
            batch = []
    if batch:
        store.upsert(batch, embedder)

def search(query_text: str,
           top_k: int = 5,
           where: Optional[Dict[str, Any]] = None,
           store: Optional[VectorStore] = None,
           embedder: Optional[EmbeddingClient] = None) -> List[Dict[str, Any]]:
    store = store or VectorStore()
    embedder = embedder or EmbeddingClient()
    out = store.query(query_text, top_k=top_k, embedder=embedder, where=where)
    # Normalize to a friendly list
    results = []
    for i in range(len(out.get("ids", [[]])[0])):
        results.append({
            "id": out["ids"][0][i],
            "text": out["documents"][0][i],
            "metadata": out["metadatas"][0][i],
            "distance": out.get("distances", [[None]])[0][i],
        })
    return results

