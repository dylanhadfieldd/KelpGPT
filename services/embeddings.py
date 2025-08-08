# services/embeddings.py
from typing import List
from config import EMBEDDING_PROVIDER, OPENAI_API_KEY, OPENAI_EMBED_MODEL, LOCAL_EMBED_MODEL

# Lazy imports to avoid heavy startup time if not needed
_client = None
_model = None

def _ensure_openai():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)

def _ensure_local():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(LOCAL_EMBED_MODEL)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns list of embedding vectors for a list of input texts.
    """
    if EMBEDDING_PROVIDER == "openai":
        _ensure_openai()
        # OpenAI can batch; keep it simple and safe
        resp = _client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    else:
        _ensure_local()
        vecs = _model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
        # Ensure plain python lists
        return [v.tolist() for v in vecs]

def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]
