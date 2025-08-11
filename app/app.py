# app.py
# Streamlit + Passcode Gate + Optional RAG (Chroma) + Chat with OpenAI
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
@@ -29,24 +30,12 @@ def _first_existing_local(*names: str) -> Optional[Path]:
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

# Image file candidates (place in SAME folder as app.py)
LOGO_FILE = _first_existing_local("logo_icon.png","logo_icon.jpg","kelp_ark_logo.png","kelp_ark_logo.jpg")
ASSISTANT_ICON_FILE = _first_existing_local("icon_kelp.png","icon_kelp.jpg","kelp_icon.png","kelp_icon.jpg")
USER_ICON_FILE      = _first_existing_local("test_tube.png","test_tube.jpg","icon_test_tube.png","icon_test_tube.jpg")
# Image files in SAME folder as app.py
LOGO_FILE = _first_existing_local("kelp_ark_logo.png","logo_icon.jpg","kelp_ark_logo.png","kelp_ark_logo.jpg")
ASSISTANT_ICON_FILE = _first_existing_local("model_avatar.png","icon_kelp.jpg","kelp_icon.png","kelp_icon.jpg")
USER_ICON_FILE      = _first_existing_local("user_model.png","test_tube.jpg","icon_test_tube.png","icon_test_tube.jpg")

# Page config (favicon/tab)
# Configure page EARLY so favicon shows on auth page too
st.set_page_config(page_title="KelpGPT", page_icon=(str(LOGO_FILE) if LOGO_FILE else "🪸"), layout="wide")

# --- Local dev only: load .env if present (ignored in git) ---
@@ -56,13 +45,11 @@ def _to_data_uri(p: Optional[Path]) -> Optional[str]:
except Exception:
pass

# ---------------------------
# Secrets / Config
# ---------------------------
# ---------- Secrets / Config ----------
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
try:
        val = st.secrets.get(name)
        return val if val is not None else os.getenv(name, default)
        v = st.secrets.get(name)
        return v if v is not None else os.getenv(name, default)
except Exception:
return os.getenv(name, default)

@@ -78,12 +65,10 @@ def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
st.error("Server missing APP_PASSCODE.")
st.stop()

# ---------------------------
# Auth Gate
# ---------------------------
# ---------- Auth Gate ----------
ATTEMPT_LIMIT = 5
LOCKOUT_SECS  = 120
SESSION_TTL   = 60 * 60
SESSION_TTL   = 60 * 60  # 1 hour

if "authed" not in st.session_state: st.session_state.authed = False
if "attempts" not in st.session_state: st.session_state.attempts = 0
@@ -92,8 +77,10 @@ def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:

def require_auth() -> bool:
now = time.time()

if st.session_state.authed and (now - st.session_state.auth_time > SESSION_TTL):
st.session_state.authed = False

if st.session_state.authed:
return True

@@ -105,6 +92,7 @@ def require_auth() -> bool:

st.title("🔒 Team Access")
st.caption("Enter the 4‑digit passcode to use KelpGPT.")

col1, col2 = st.columns([3,1])
with col1:
code = st.text_input("Passcode", type="password", max_chars=4, help="Ask Dylan for the current code.")
@@ -126,73 +114,39 @@ def require_auth() -> bool:
left = ATTEMPT_LIMIT - st.session_state.attempts
st.error(f"Incorrect passcode. {left} attempt(s) left.")
st.stop()

st.stop()

if not require_auth():
st.stop()

# ---------------------------
# OpenAI client
# ---------------------------
# ---------- OpenAI client ----------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Chroma: adaptive open (handles 1536 vs 384)
# ---------------------------
import chromadb
from chromadb.errors import InvalidArgumentError

# ---------- Chroma (matches ingest.py: OpenAI 1536-dim) ----------
@st.cache_resource(show_spinner=False)
def _open_collection_with_openai():
def _get_chroma():
    import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.OpenAIEmbeddingFunction(
api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",  # 1536 dims
        model_name="text-embedding-3-small",  # 1536-dim
)
cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = cli.get_or_create_collection(name=TEXT_COLLECTION, embedding_function=ef)
return cli, col

@st.cache_resource(show_spinner=False)
def _open_collection_as_is():
    cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = cli.get_collection(name=TEXT_COLLECTION)
    return cli, col

def _get_chroma(prefer_openai_first: bool = True):
    if prefer_openai_first:
        try:
            return _open_collection_with_openai()
        except Exception:
            # fall back to opening existing collection without embedding fn
            return _open_collection_as_is()
    else:
        try:
            return _open_collection_as_is()
        except Exception:
            return _open_collection_with_openai()

def _rag_mode_label(col) -> str:
    # best-effort label
    try:
        md = col.metadata or {}
        model = md.get("embedding_model") or md.get("hnsw:space") or "unknown"
        return str(model)
    except Exception:
        return "unknown"

# ---------------------------
# Utilities for PDFs & APA
# ---------------------------
# ---------- PDF helpers / APA ----------
def _chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
words = text.split()
chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end - overlap > start else end
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if j - overlap > i else j
return chunks

def _parse_pdf_year(raw_date: Optional[str]) -> Optional[str]:
@@ -213,59 +167,31 @@ def _extract_pdf_core_metadata(file_bytes: bytes) -> Dict[str, Any]:
except Exception:
return {"title": None, "authors": None, "year": None}

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

def _build_apa_citation(meta: Dict[str, Any], fallback_filename: str) -> str:
    authors = meta.get("apa_authors") or meta.get("authors")
    year    = meta.get("apa_year") or meta.get("year")
    title   = meta.get("apa_title") or meta.get("title")
    journal = meta.get("apa_container") or meta.get("journal")
    doi     = meta.get("apa_doi")
    url     = meta.get("url")
    authors = meta.get("authors")
    year    = meta.get("year")
    title   = meta.get("title")
if not (authors or title):
return fallback_filename
parts = []
if authors: parts.append(f"{authors}")
if year:    parts.append(f"({year}).")
if title:   parts.append(f"{title}.")
    if journal: parts.append(f"*{journal}*.")
    if doi: parts.append(f"https://doi.org/{doi}")
    elif url: parts.append(f"{url}")
return " ".join(parts).strip()

def _dedupe_preserve_order(items: List[str]) -> List[str]:
seen, out = set(), []
for x in items:
if x not in seen:
            seen.add(x)
            out.append(x)
            seen.add(x); out.append(x)
return out

# ---------------------------
# Sidebar / Controls
# ---------------------------
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
    if LOGO_FILE:
        st.logo(str(LOGO_FILE))
    else:
        st.write("KelpGPT")

st.header("KelpGPT")
st.caption("Internal research assistant")
@@ -277,91 +203,84 @@ def _dedupe_preserve_order(items: List[str]) -> List[str]:
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
                _, collection = _get_chroma(prefer_openai_first=True)
                _, collection = _get_chroma()
add_count = 0
for f in up_files:
raw = f.read()
                    text = _extract_text_from_pdf(raw)
                    # Extract text
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(io.BytesIO(raw))
                        texts = []
                        for page in reader.pages:
                            try:
                                t = page.extract_text() or ""
                            except Exception:
                                t = ""
                            if t: texts.append(t)
                        text = "\n\n".join(texts)
                    except Exception:
                        text = ""
if not text.strip():
continue
chunks = _chunk_text(text)

                    # Minimal metadata
core = _extract_pdf_core_metadata(raw)
                    user = sidecar_map.get(f.name, {})
base_meta = {
"source_filename": f.name,
                        "apa_authors": user.get("apa_authors"),
                        "apa_year": user.get("apa_year"),
                        "apa_title": user.get("apa_title"),
                        "apa_container": user.get("apa_container"),
                        "apa_doi": user.get("apa_doi"),
                        "url": user.get("url"),
"authors": core.get("authors"),
"year": core.get("year"),
"title": core.get("title"),
                        "journal": user.get("apa_container"),
}

                    # Content-addressable IDs
import hashlib
                    base = hashlib.sha1(raw).hexdigest()
                    ids = [f"{base}-{i}" for i in range(len(chunks))]
                    digest = hashlib.sha1(raw).hexdigest()
                    ids = [f"{digest}-{i}" for i in range(len(chunks))]
metadatas = []
for i in range(len(chunks)):
m = dict(base_meta)
m["chunk"] = i
metadatas.append(m)

collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
add_count += len(chunks)

st.success(f"Ingested {add_count} chunks.")
                if meta_sidecar is not None:
                    st.info("Metadata JSON applied where filenames matched.")

# ---------------------------
# Header (HTML with data URIs so images render)
# ---------------------------
LOGO_DATA_URI = _to_data_uri(LOGO_FILE)
def _mk_header_html(logo_data_uri: Optional[str]) -> str:
    left = f'<img src="{logo_data_uri}" style="width:40px;height:40px;border-radius:8px;object-fit:cover;">' if logo_data_uri else ''
    right = f'<img src="{logo_data_uri}" alt="Kelp Ark" style="height:40px;">' if logo_data_uri else ''
    return dedent(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin:6px 0 10px 0;">
      <div style="display:flex;align-items:center;gap:12px;">
        {left}
        <div>
          <div style="font-size:22px;font-weight:600;line-height:1.1;">I'm KARA, how can I help you?</div>
          <div style="margin-top:2px;color:#8a8a8a;">KelpArk Research Assistant</div>
        </div>
      </div>
      <div>{right}</div>
    </div>
    """)

st.markdown(_mk_header_html(LOGO_DATA_URI), unsafe_allow_html=True)

# ---------------------------
# RAG sanity
# ---------------------------

    st.markdown("---")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    st.session_state["model"] = model
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.3, 0.1)
    st.session_state["temperature"] = temperature

# ---------- Top header (logo right) ----------
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
    _, _col_probe = _get_chroma(prefer_openai_first=True)
    st.info(f"RAG online. Collection '{TEXT_COLLECTION}' contains ~{_col_probe.count()} chunks.")
    _, _col = _get_chroma()
    st.info(f"RAG online. Collection '{TEXT_COLLECTION}' contains ~{_col.count()} chunks.")
except Exception as e:
st.warning(f"RAG not available: {e}")

# ---------------------------
# Avatars + Chat history
# ---------------------------
# ---------- Avatars + history ----------
AVATAR_ASSISTANT = (str(ASSISTANT_ICON_FILE) if ASSISTANT_ICON_FILE else "🪸")
AVATAR_USER      = (str(USER_ICON_FILE)      if USER_ICON_FILE      else "🧪")

@@ -376,30 +295,20 @@ def _mk_header_html(logo_data_uri: Optional[str]) -> str:
with st.chat_message(m["role"], avatar=avatar):
st.markdown(m["content"])

# ---------------------------
# Retrieval helpers (with adaptive retry)
# ---------------------------
def _get_collection(prefer_openai_first: bool = True):
# ---------- Retrieval ----------
def _get_collection():
try:
        _, c = _get_chroma(prefer_openai_first=prefer_openai_first)
        return c
        _, collection = _get_chroma()
        return collection
except Exception as e:
st.warning(f"RAG not available: {e}")
return None

def retrieve(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    col = _get_collection(prefer_openai_first=True)
    if col is None:
    collection = _get_collection()
    if collection is None:
return []
    try:
        res = col.query(query_texts=[query], n_results=k)
    except InvalidArgumentError as e:
        # Likely a 1536 vs 384 mismatch; reopen with the other mode and retry once
        col = _get_collection(prefer_openai_first=False)
        if col is None:
            st.warning(f"RAG failed: {e}")
            return []
        res = col.query(query_texts=[query], n_results=k)
    res = collection.query(query_texts=[query], n_results=k)
docs = res.get("documents", [[]])[0]
metas = res.get("metadatas", [[]])[0]
return list(zip(docs, metas))
@@ -424,13 +333,8 @@ def build_apa_sources_note(ctx: List[Tuple[str, dict]]) -> str:
apa_lines = _dedupe_preserve_order(apa_lines)
return "\n\n**References**\n" + "\n".join(f"- {line}" for line in apa_lines)

# ---------------------------
# Chat submit
# ---------------------------
model = st.session_state.get("model") or "gpt-4o-mini"  # from selectbox; keep safe default
temperature = st.session_state.get("temperature") or 0.3
# ---------- Chat submit ----------
prompt = st.chat_input("Ask something…")

if prompt:
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user", avatar=AVATAR_USER):
@@ -459,19 +363,18 @@ def build_apa_sources_note(ctx: List[Tuple[str, dict]]) -> str:
with st.spinner("Thinking…"):
try:
resp = client.chat.completions.create(
                    model=model,
                    model=st.session_state["model"],
messages=convo_msgs,
                    temperature=temperature,
                    temperature=st.session_state["temperature"],
)
answer = resp.choices[0].message.content
except Exception as e:
answer = f"Error from model: {e}"

st.markdown(answer + (("\n\n" + refs_block) if refs_block else ""))

st.session_state.messages.append({"role": "assistant", "content": answer + (("\n\n" + refs_block) if refs_block else "")})

# ---------------------------
# Footer
# ---------------------------
# ---------- Footer ----------
st.markdown("---")
st.caption("Tip: Rotate the passcode from Streamlit **Secrets** when needed. Documents you upload are stored in the app server's Chroma DB and never pushed to Git.")
st.caption("Tip: Rotate the passcode in Streamlit **Secrets** when needed. PDFs are stored in the app server's Chroma DB and never pushed to Git.")
