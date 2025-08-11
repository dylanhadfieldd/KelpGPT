# app.py
# ---------------------------------------------------------------
# Streamlit + Passcode Gate + Optional RAG (Chroma) + Chat with OpenAI
# - Prioritizes matches to paper_title/authors (e.g., "Jose's paper")
# - Citations show human-readable paper titles (not filenames)
# - Windows-safe paths and environment handling
# ---------------------------------------------------------------

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st

# --- Images next to app.py (Windows-safe) ---
LOGO_PATH = os.getenv("LOGO_PATH", "logo_icon.jpg")
ICON_PATH = os.getenv("ICON_PATH", "icon.jpg")  # can be .png/.jpg as long as file exists

st.set_page_config(page_title="KelpGPT", page_icon=LOGO_PATH, layout="wide")

# --- Local dev only: load .env if present ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---- OpenAI (Responses + Embeddings) ----
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"

client_oa = OpenAI(api_key=OPENAI_API_KEY)

# ---- Chroma settings ----
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION    = os.getenv("TEXT_COLLECTION", "papers_text")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-large")  # must match ingest
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "6"))

# ---- Access Gate ----
APP_PASSCODE = os.getenv("APP_PASSCODE", "")

# ---- UI Helpers ----
def _file_exists(p: str) -> bool:
    try:
        return Path(p).is_file()
    except Exception:
        return False

def _first_existing(*names: str) -> str:
    for n in names:
        if n and _file_exists(n):
            return n
    return ""

# ---- Chroma (lazy init) ----
@st.cache_resource(show_spinner=False)
def _chroma_and_collection():
    chroma = Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PERSIST_DIR,
    ))
    # Attach the same embedding function used at creation
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL
    )
    coll = chroma.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"embedding_model": EMBED_MODEL},
        embedding_function=ef
    )
    # Guard against model mismatch
    meta = coll.metadata or {}
    used = meta.get("embedding_model", "")
    if used and used != EMBED_MODEL:
        st.error(
            f"Chroma collection '{TEXT_COLLECTION}' was created with '{used}', "
            f"but this app is configured for '{EMBED_MODEL}'. "
            f"Delete the folder '{CHROMA_PERSIST_DIR}' or reconfigure EMBED_MODEL."
        )
    return chroma, coll

# ---- Retrieval & Re-ranking ----
def _simple_normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def boost_score(query: str, meta: Dict[str, Any], base_dist: float) -> float:
    """
    Chroma returns cosine distance (smaller = better).
    We transform to a score and apply boosts when the query matches paper_title/authors.
    """
    q = _simple_normalize(query)
    title = _simple_normalize(meta.get("paper_title", ""))
    authors = _simple_normalize(meta.get("authors", ""))
    fname = _simple_normalize(meta.get("file_name", ""))

    # Convert distance to a base score
    base_score = 1.0 / (1.0 + max(base_dist, 1e-6))

    boost = 1.0
    # Prioritize exact-ish hits (e.g., "jose", "cruz-burgos", actual title words)
    if any(tok and tok in title for tok in q.split()):
        boost *= 1.5
    if any(tok and tok in authors for tok in q.split()):
        boost *= 1.4
    # If user literally mentioned the filename, give a small nudge
    if any(tok and tok in fname for tok in q.split()):
        boost *= 1.1

    return base_score * boost

def retrieve_with_boost(query: str, k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    """
    Query Chroma and re-rank by metadata/title/author match so that
    'Jose's paper' pulls from that document first.
    """
    _, coll = _chroma_and_collection()
    if coll.count() == 0:
        return []

    try:
        res = coll.query(query_texts=[query], n_results=max(k * 2, 8), include=["distances", "metadatas", "documents"])
    except Exception as e:
        st.warning(f"RAG not available: {e}")
        return []

    results = []
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        score = boost_score(query, meta, dist)
        results.append({
            "text": doc,
            "meta": meta,
            "dist": float(dist),
            "score": float(score),
        })

    # sort by boosted score (high -> low)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]

def build_context_and_citations(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Combine top chunks and collect unique paper titles for display."""
    context_parts = []
    titles = []
    seen_docs = set()
    for r in results:
        context_parts.append(r["text"])
        t = r["meta"].get("paper_title") or r["meta"].get("file_name")
        if t and t not in titles:
            titles.append(t)
        # keep 1-2 chunks per doc to avoid swamping
        did = r["meta"].get("doc_id")
        if did:
            seen_docs.add(did)
        if len(context_parts) >= 6:
            break
    return "\n\n---\n\n".join(context_parts), titles

# ---- OpenAI chat call ----
def chat_with_openai(prompt: str, system: str) -> str:
    resp = client_oa.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ---- UI ----
def auth_gate() -> bool:
    if not APP_PASSCODE:
        return True
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return True

    with st.sidebar:
        st.subheader("🔐 Team Access")
        code = st.text_input("Enter passcode", type="password")
        if st.button("Unlock"):
            if code == APP_PASSCODE:
                st.session_state.authed = True
                st.success("Access granted")
            else:
                st.error("Incorrect code")
    return st.session_state.authed

def main():
    st.sidebar.image(_first_existing(LOGO_PATH), width=140)
    st.sidebar.markdown("### KelpGPT – Research Assistant")
    use_rag = st.sidebar.toggle("Use local RAG (Chroma)", value=True)
    st.sidebar.caption(f"DB: `{Path(CHROMA_PERSIST_DIR).resolve()}`")

    st.title("KelpGPT")
    st.write("Ask about **specific papers** (e.g., *Jose’s paper*) or general topics. "
             "When RAG is on, answers prioritize your ingested PDFs.")

    if not auth_gate():
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat input
    user_msg = st.chat_input("Ask me anything about your papers…")
    if user_msg:
        st.session_state.history.append(("user", user_msg))

        retrieved = []
        citations = []
        context = ""
        if use_rag:
            retrieved = retrieve_with_boost(user_msg, RAG_TOP_K)
            if retrieved:
                context, citations = build_context_and_citations(retrieved)

        # Build the system prompt
        sys_prompt = (
            "You are KelpGPT, a marine science research assistant. "
            "If context is provided, answer **primarily** from it. "
            "Cite papers at the end as a bulleted list using the paper titles from metadata. "
            "Only bring in general knowledge if the context is insufficient, and say so."
        )

        final_prompt = user_msg
        if context:
            final_prompt = (
                "Use the following document context first.\n\n"
                f"### CONTEXT START\n{context}\n### CONTEXT END\n\n"
                f"Question: {user_msg}"
            )

        with st.spinner("Thinking…"):
            answer = chat_with_openai(final_prompt, sys_prompt)

        st.session_state.history.append(("assistant", answer, citations))

    # Render chat
    for role_tuple in st.session_state.history:
        role = role_tuple[0]
        if role == "user":
            with st.chat_message("user"):
                st.write(role_tuple[1])
        else:
            # assistant
            msg = role_tuple[1]
            cites = role_tuple[2] if len(role_tuple) > 2 else []
            with st.chat_message("assistant", avatar=_first_existing(ICON_PATH, LOGO_PATH)):
                st.write(msg)
                if cites:
                    st.markdown("**Sources:**")
                    for t in cites:
                        st.markdown(f"- {t}")

    st.caption("Tip: If you say *“Tell me about Jose’s paper”*, the retriever boosts documents whose "
               "title or author metadata match ‘Jose’ so those chunks rank first.")

if __name__ == "__main__":
    main()
