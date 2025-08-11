# app.py
# Streamlit + Passcode Gate + RAG (Chroma) + Chat with OpenAI
# ---------------------------------------------------------------
# - Uses OpenAI embeddings (1536-dim) to match ingest.py
# - Kelp Ark logo: favicon + top-right header + sidebar
# - Assistant avatar: kelp icon; User avatar: test tube icon
# - Optional upload + ingest inside the app
# ---------------------------------------------------------------

import os
import io
import re
import time
import json
import base64
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from pathlib import Path
from PIL import Image
from textwrap import dedent

# ---------- Paths & Helpers ----------
ROOT = Path(__file__).parent.resolve()

# ---------------------------
# Image helpers
# ---------------------------
def _first_existing_local(*names: str) -> Optional[Path]:
    for n in names:
        if not n:
            continue
        p = (ROOT / n).resolve()
        if p.exists():
            return p
    return None

def _to_data_uri(p: Optional[Path]) -> Optional[str]:
    if not p:
        return None
    try:
        raw = p.read_bytes()
        ext = p.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

# Image files in SAME folder as app.py
LOGO_FILE = _first_existing_local("logo_icon.png","logo_icon.jpg","kelp_ark_logo.png","kelp_ark_logo.jpg")
ASSISTANT_ICON_FILE = _first_existing_local("icon_kelp.png","icon_kelp.jpg","kelp_icon.png","kelp_icon.jpg","model_avatar.png")
USER_ICON_FILE      = _first_existing_local("test_tube.png","test_tube.jpg","icon_test_tube.png","icon_test_tube.jpg","user_model.png")

# Configure page EARLY so favicon shows on auth page too
st.set_page_config(page_title="KelpGPT", page_icon=(str(LOGO_FILE) if LOGO_FILE else "🪸"), layout="wide")

# --- Local dev only: load .env if present (ignored in git) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Secrets / Config ----------
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        return v if v is not None else os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY     = _get_secret("OPENAI_API_KEY", "")
APP_PASSCODE       = _get_secret("APP_PASSCODE", "")
CHROMA_PERSIST_DIR = _get_secret("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION    = _get_secret("TEXT_COLLECTION", "papers_text")
EMBED_MODEL        = _get_secret("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim default

if not OPENAI_API_KEY:
    st.error("Server missing OPENAI_API_KEY.")
    st.stop()
if not APP_PASSCODE:
    st.error("Server missing APP_PASSCODE.")
    st.stop()

# ---------- Auth Gate ----------
ATTEMPT_LIMIT = 5
LOCKOUT_SECS  = 120
SESSION_TTL   = 60 * 60  # 1 hour

if "authed" not in st.session_state: st.session_state.authed = False
if "attempts" not in st.session_state: st.session_state.attempts = 0
if "lockout_until" not in st.session_state: st.session_state.lockout_until = 0
if "auth_time" not in st.session_state: st.session_state.auth_time = 0
if "messages" not in st.session_state: st.session_state.messages = []

def require_auth() -> bool:
    now = time.time()
    if st.session_state.authed and (now - st.session_state.auth_time > SESSION_TTL):
        st.session_state.authed = False

    if st.session_state.authed:
        return True

    if now < st.session_state.lockout_until:
        wait = int(st.session_state.lockout_until - now)
        st.error(f"Locked out. Try again in {wait}s.")
        st.stop()

    st.title("🔒 Team Access")
    st.caption("Enter the 4‑digit passcode to use KelpGPT.")

    col1, col2 = st.columns([3,1])
    with col1:
        code = st.text_input("Passcode", type="password", max_chars=4, help="Ask Dylan for the current code.")
    with col2:
        submit = st.button("Enter")

    if submit:
        if code == APP_PASSCODE:
            st.session_state.authed = True
            st.session_state.auth_time = time.time()
            st.session_state.attempts = 0
            st.rerun()
        else:
            st.session_state.attempts += 1
            if st.session_state.attempts >= ATTEMPT_LIMIT:
                st.session_state.lockout_until = time.time() + LOCKOUT_SECS
                st.error("Too many attempts. Locked temporarily.")
                st.stop()
            left = ATTEMPT_LIMIT - st.session_state.attempts
            st.error(f"Incorrect passcode. {left} attempt(s) left.")
            st.stop()

    st.stop()

if not require_auth():
    st.stop()

# ---------- OpenAI client ----------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Chroma (matches ingest.py) ----------
import chromadb
from chromadb.utils import embedding_functions

@st.cache_resource(show_spinner=False)
def _get_chroma():
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,  # keep in sync with ingest.py
    )
    cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = cli.get_or_create_collection(name=TEXT_COLLECTION, metadata={"embedding_model": EMBED_MODEL}, embedding_function=ef)
    return cli, col

def _get_collection():
    try:
        _, collection = _get_chroma()
        return collection
    except Exception as e:
        st.warning(f"RAG not available: {e}")
        return None

# ---------- PDF helpers / APA ----------
def _chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks

def _extract_pdf_core_metadata(file_bytes: bytes) -> Dict[str, Any]:
    meta = {"title": None, "authors": None, "year": None, "journal": None}
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        info = reader.metadata or {}
        t = str(info.get("/Title", "") or info.get("Title", "") or "").strip()
        a = str(info.get("/Author","") or info.get("Author","") or "").strip()
        y = str(info.get("/CreationDate","") or info.get("CreationDate","") or "")
        year = None
        m = re.search(r"D:\d{4}", y) or re.search(r"\b(19|20)\d{2}\b", y)
        if m:
            year = re.search(r"(19|20)\d{2}", m.group(0)).group(0)
        meta.update({"title": t or None, "authors": a or None, "year": year})
    except Exception:
        pass
    return meta

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n\n".join(texts)

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _build_apa_line(meta: Dict[str, Any], fallback_filename: str) -> str:
    # Prefer ingest.py keys, then inline-app ingest keys
    authors = meta.get("authors")
    title   = meta.get("paper_title") or meta.get("title")
    year    = meta.get("year")
    if not (authors or title):
        return fallback_filename
    parts = []
    if authors: parts.append(f"{authors}")
    if year:    parts.append(f"({year}).")
    if title:   parts.append(f"{title}.")
    return " ".join(parts).strip()

def build_apa_sources_note(ctx: List[Tuple[str, dict]]) -> str:
    apa_lines = []
    for _, m in ctx:
        line = _build_apa_line(m or {}, m.get("file_name") or m.get("source_filename") or "Unknown source")
        apa_lines.append(f"- {line}")
    apa_lines = _dedupe_preserve_order(apa_lines)
    return "\n\n**References**\n" + "\n".join(apa_lines) if apa_lines else ""

# ---------- Sidebar ----------
with st.sidebar:
    try:
        if LOGO_FILE:
            st.logo(str(LOGO_FILE))
        else:
            st.write("KelpGPT")
    except Exception:
        if LOGO_FILE:
            st.image(str(LOGO_FILE), use_container_width=True)
        else:
            st.write("KelpGPT")

st.header("KelpGPT")
st.caption("Internal research assistant")
st.markdown("---")

# Model + temperature controls
model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
st.session_state["model"] = model
temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.3, 0.1)
st.session_state["temperature"] = temperature

# ---------- Top header (logo on right) ----------
left, right = st.columns([1, 0.12])
with left:
    st.markdown(
        "<div style='font-size:22px;font-weight:600;line-height:1.1;'>I'm KARA, how can I help you?</div>"
        "<div style='margin-top:2px;color:#8a8a8a;'>KelpArk Research Assistant</div>",
        unsafe_allow_html=True
    )
with right:
    if LOGO_FILE:
        st.image(str(LOGO_FILE), width=64)

# ---------- RAG sanity ----------
try:
    col_probe = _get_collection()
    if col_probe is not None:
        st.info(f"RAG online. Collection '{TEXT_COLLECTION}' contains ~{col_probe.count()} chunks.")
    else:
        st.warning("RAG not available.")
except Exception as e:
    st.warning(f"RAG not available: {e}")

# ---------- Upload & Ingest (optional inside app) ----------
st.markdown("---")
st.subheader("📥 Upload PDFs (optional)")
up_files = st.file_uploader("Add PDFs to the local index (Chroma).", type=["pdf"], accept_multiple_files=True)
meta_sidecar = st.file_uploader("Optional: metadata JSON (APA fields per filename)", type=["json"])

if st.button("Ingest PDFs"):
    if not up_files:
        st.warning("No PDFs selected.")
    else:
        sidecar_map: Dict[str, Dict[str, Any]] = {}
        if meta_sidecar is not None:
            try:
                sidecar_map = json.loads(meta_sidecar.read().decode("utf-8"))
            except Exception as e:
                st.warning(f"Could not parse metadata JSON: {e}")

        with st.spinner("Ingesting…"):
            collection = _get_collection()
            if collection is None:
                st.error("RAG not available.")
            else:
                add_count = 0
                for f in up_files:
                    raw = f.read()
                    text = _extract_text_from_pdf(raw)
                    if not text.strip():
                        continue
                    chunks = _chunk_text(text)

                    core = _extract_pdf_core_metadata(raw)
                    user = sidecar_map.get(f.name, {})
                    base_meta = {
                        "source_filename": f.name,
                        "authors": user.get("apa_authors") or core.get("authors"),
                        "year":    user.get("apa_year")    or core.get("year"),
                        "paper_title": user.get("apa_title") or core.get("title"),
                        "journal": user.get("apa_container"),
                        "url": user.get("url"),
                    }

                    import hashlib
                    digest = hashlib.sha1(raw).hexdigest()
                    ids = [f"{digest}-{i}" for i in range(len(chunks))]
                    metadatas = []
                    for i in range(len(chunks)):
                        m = dict(base_meta)
                        m["chunk_index"] = i
                        metadatas.append(m)

                    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
                    add_count += len(chunks)

                st.success(f"Ingested {add_count} chunks.")
                if meta_sidecar is not None:
                    st.info("Metadata JSON applied where filenames matched.")

# ---------- Avatars + history ----------
AVATAR_ASSISTANT = (str(ASSISTANT_ICON_FILE) if ASSISTANT_ICON_FILE else "🪸")
AVATAR_USER      = (str(USER_ICON_FILE)      if USER_ICON_FILE      else "🧪")

for m in st.session_state.messages:
    avatar = AVATAR_USER if m["role"] == "user" else AVATAR_ASSISTANT
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# ---------- Retrieval ----------
def retrieve(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    collection = _get_collection()
    if collection is None:
        return []
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

# ---------- Chat submit ----------
prompt = st.chat_input("Ask something…")

def _build_context_block(ctx: List[Tuple[str, dict]]) -> str:
    if not ctx:
        return ""
    lines = []
    for i, (doc, m) in enumerate(ctx, 1):
        title = m.get("paper_title") or m.get("title") or m.get("source_filename") or "Unknown source"
        lines.append(f"[{i}] {title} — chunk {m.get('chunk_index', '?')}")
        lines.append(doc)
        lines.append("")
    return "\n".join(lines).strip()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    # Retrieve
    ctx = retrieve(prompt, k=6)
    context_block = _build_context_block(ctx)
    refs_block = build_apa_sources_note(ctx)

    # Build conversation payload
    convo_msgs = []
    convo_msgs.append({
        "role": "system",
        "content": (
            "You are KelpGPT, a marine science research assistant for Kelp Ark. "
            "Prefer information from the provided context. If the user asks about a specific paper (e.g., 'Jose's paper'), "
            "answer primarily from the most relevant chunks whose metadata paper_title matches. "
            "Be concise and factual. If context is missing, say so and proceed with best knowledge."
        )
    })
    if context_block:
        convo_msgs.append({
            "role": "system",
            "content": f"Context (top-k retrieved, cite by paper title in a References section):\n\n{context_block}"
        })
    # include prior turns for style continuity (no contexts)
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant"):
            convo_msgs.append(m)

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("Thinking…"):
            try:
                resp = client.chat.completions.create(
                    model=st.session_state["model"],
                    messages=convo_msgs,
                    temperature=st.session_state["temperature"],
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Error from model: {e}"

            st.markdown(answer + (("\n\n" + refs_block) if refs_block else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer + (("\n\n" + refs_block) if refs_block else "")})

# ---------- Footer ----------
st.markdown("---")
st.caption("Tip: Rotate the passcode in Streamlit **Secrets** when needed. PDFs are stored in the app server's Chroma DB and never pushed to Git.")
