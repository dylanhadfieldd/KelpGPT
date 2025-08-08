# services/figure_index.py
import chromadb
from typing import Dict, List
from config import CHROMA_PATH, FIGURES_COLLECTION
from services.embeddings import embed_texts

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_coll = _client.get_or_create_collection(FIGURES_COLLECTION, metadata={"hnsw:space": "cosine"})

def upsert_figures(records: List[Dict]):
    """
    records: [{ id, caption, surrounding_text, img_url, sha256, doc_id, doc_title, page, bbox?, width?, height? }]
    Embeds caption+context and upserts.
    Dedup by sha256: skip if existing item has same sha256 (best-effort).
    """
    if not records:
        return

    # Soft-dedup: query existing sha256s
    metas = _coll.get(include=["metadatas"])
    existing_hashes = set()
    if metas and metas.get("metadatas"):
        for md in metas["metadatas"]:
            if isinstance(md, dict) and md.get("sha256"):
                existing_hashes.add(md["sha256"])

    new_records = [r for r in records if r.get("sha256") not in existing_hashes]
    if not new_records:
        return

    texts = []
    ids = []
    metadatas = []
    for r in new_records:
        cap = (r.get("caption") or "").strip()
        ctx = (r.get("surrounding_text") or "").strip()
        text_for_embed = f"{cap}\n\n{ctx}".strip() or f"{r.get('doc_title','')} page {r.get('page','')}"
        texts.append(text_for_embed)
        ids.append(r["id"])
        metadatas.append({
            "caption": cap,
            "surrounding_text": ctx,
            "img_url": r.get("img_url"),
            "sha256": r.get("sha256"),
            "doc_id": r.get("doc_id"),
            "doc_title": r.get("doc_title"),
            "page": r.get("page"),
            "bbox": r.get("bbox"),
            "width": r.get("width"),
            "height": r.get("height"),
        })

    embeddings = embed_texts(texts)
    _coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

def search_figures(query: str, k: int = 12) -> List[Dict]:
    """
    Returns [{caption, img_url, doc_id, doc_title, page, score}]
    """
    from services.embeddings import embed_text
    qvec = embed_text(query)
    res = _coll.query(query_embeddings=[qvec], n_results=k, include=["metadatas", "distances", "ids"])
    out = []
    if res and res.get("metadatas"):
        metas = res["metadatas"][0]
        dists = res.get("distances", [[ ]])[0] or []
        ids = res.get("ids", [[]])[0] or []
        for m, d, id_ in zip(metas, dists, ids):
            out.append({
                "id": id_,
                "caption": m.get("caption", ""),
                "img_url": m.get("img_url", ""),
                "doc_id": m.get("doc_id", ""),
                "doc_title": m.get("doc_title", ""),
                "page": m.get("page", ""),
                "score": 1 - d if isinstance(d, (int, float)) else None,  # cosine sim
            })
    return out
