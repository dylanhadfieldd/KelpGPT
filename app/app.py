# app.py
# Streamlit + Passcode Gate + Optional RAG (Chroma) + Chat with OpenAI
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

ROOT = Path(__file__).parent.resolve()

# ---------------------------
# Image helpers
# ---------------------------
def _first_existing_local(*names: str) -> Optional[Path]:
    for n in names:
        if not n:
            continue
        p = (ROOT / n).resolve()
        if p.exists() and p.is_file():
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

# Page config (favicon/tab)
st.set_page_config(page_title="KelpGPT", page_icon=(str(LOGO_FILE) if LOGO_FILE else "🪸"), layout="wide")

# --- Local dev only: load .env if present (ignored in git) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------
# Secrets / Config
# ---------------------------
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        val = st.secrets.get(name)
        return val if val is not None else os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY     = _get_secret("OPENAI_API_KEY")
APP_PASSCODE       = _get_secret("APP_PASSCODE")
CHROMA_PERSIST_DIR = _get_secret("CHROMA_PERSIST_DIR", "chroma_db")
TEXT_COLLECTION    = _get_secret("TEXT_COLLECTION", "papers_text")

if not OPENAI_API_KEY:
    st.error("Server missing OPENAI_API_KEY.")
    st.stop()
if not APP_PASSCODE:
    st.error("Server missing APP_PASSCODE.")
    st.stop()

# ---------------------------
# Auth Gate
# ---------------------------
ATTEMPT_LIMIT = 5
LOCKOUT_SECS  = 120
SESSION_TTL   = 60 * 60

if "authed" not in st.session_state: st.session_state.authed = False
if "attempts" not in st.session_state: st.session_state.attempts = 0
if "lockout_until" not in st.session_state: st.session_state.lockout_until = 0.0
if "auth_time" not in st.session_state: st.session_state.auth_time = 0.0

def require_auth() -> bool:
    now = time.time()
    if st.session_state.authed and (now - st.session_state.auth_time > SESSION_TTL):
        st.session_state.authed = False
    if st.session_state.authed:
        return True

    if now < st.session_state.lockout_until:
        remaining = int(st.session_state.lockout_until - now)
        st.title("🔒 Team Access")
        st.warning(f"Too many attempts. Try again in {remaining} seconds.")
        st.stop()

    st.title("🔒 Team Access")
    st.caption("Enter the 4‑digit passcode to use KelpGPT.")
    col1, col2 = st.columns([3,1])
    with col1:
        code = st.text_input("Passcode", type="password", max_chars=4, help="Ask Dylan for the current code.")
    with col2:
        login = st.button("Unlock", use_container_width=True)

    if login:
        if code == APP_PASSCODE:
            st.session_state.authed = True
            st.session_state.auth_time = now
            st.session_state.attempts = 0
            return True
        else:
            st.session_state.attempts += 1
            if st.session_state.attempts >= ATTEMPT_LIMIT:
                st.session_state.lockout_until = now + LOCKOUT_SECS
                st.error("Too many incorrect attempts. Temporarily locked.")
            else:
                left = ATTEMPT_LIMIT - st.session_state.attempts
                st.error(f"Incorrect passcode. {left} attempt(s) left.")
            st.stop()
    st.stop()

if not require_auth():
    st.stop()

# ---------------------------
# OpenAI client
# ---------------------------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Chroma: adaptive open (handles 1536 vs 384)
# ---------------------------
import chromadb
from chromadb.errors import InvalidArgumentError

@st.cache_resource(show_spinner=False)
def _open_collection_with_openai():
    from chromadb.utils import embedding_functions
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",  # 1536 dims
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
def _chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end - overlap > start else end
    return chunks

def _parse_pdf_year(raw_date: Optional[str]) -> Optional[str]:
    if not raw_date:
        return None
    m = re.search(r"(19|20)\d{2}", raw_date)
    return m.group(0) if m else None

def _extract_pdf_core_metadata(file_bytes: bytes) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        md = reader.metadata or {}
        title = (getattr(md, "title", None) or md.get("/Title") or "").strip() or None
        authors = (getattr(md, "author", None) or md.get("/Author") or "").strip() or None
        year = _parse_pdf_year(getattr(md, "creation_date", None) or md.get("/CreationDate"))
        return {"title": title, "authors": authors, "year": year}
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
    return out

# ---------------------------
# Sidebar / Controls
# ---------------------------
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

    use_rag = st.toggle("Use document retrieval (RAG)", value=True,
                        help="Search your uploaded/ingested PDFs for relevant context.")
    st.session_state["use_rag"] = use_rag

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
                    import hashlib
                    base = hashlib.sha1(raw).hexdigest()
                    ids = [f"{base}-{i}" for i in range(len(chunks))]
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
try:
    _, _col_probe = _get_chroma(prefer_openai_first=True)
    st.info(f"RAG online. Collection '{TEXT_COLLECTION}' contains ~{_col_probe.count()} chunks.")
except Exception as e:
    st.warning(f"RAG not available: {e}")

# ---------------------------
# Avatars + Chat history
# ---------------------------
AVATAR_ASSISTANT = (str(ASSISTANT_ICON_FILE) if ASSISTANT_ICON_FILE else "🪸")
AVATAR_USER      = (str(USER_ICON_FILE)      if USER_ICON_FILE      else "🧪")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are KelpGPT, a precise, helpful marine science research assistant. Cite sources if provided in context."}
    ]

for m in st.session_state.messages:
    if m["role"] in ("user", "assistant"):
        avatar = AVATAR_USER if m["role"] == "user" else AVATAR_ASSISTANT
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

# ---------------------------
# Retrieval helpers (with adaptive retry)
# ---------------------------
def _get_collection(prefer_openai_first: bool = True):
    try:
        _, c = _get_chroma(prefer_openai_first=prefer_openai_first)
        return c
    except Exception as e:
        st.warning(f"RAG not available: {e}")
        return None

def retrieve(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    col = _get_collection(prefer_openai_first=True)
    if col is None:
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
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

def format_context(ctx: List[Tuple[str, dict]]) -> str:
    if not ctx:
        return ""
    lines = []
    for i, (doc, meta) in enumerate(ctx, 1):
        src = meta.get("source_filename", "doc")
        chunk = meta.get("chunk", i)
        lines.append(f"[{i}] ({src} • chunk {chunk})\n{doc}")
    return "\n\n".join(lines)

def build_apa_sources_note(ctx: List[Tuple[str, dict]]) -> str:
    if not ctx:
        return ""
    apa_lines = []
    for _, m in ctx:
        apa = _build_apa_citation(m, m.get("source_filename", "Unknown source"))
        apa_lines.append(apa)
    apa_lines = _dedupe_preserve_order(apa_lines)
    return "\n\n**References**\n" + "\n".join(f"- {line}" for line in apa_lines)

# ---------------------------
# Chat submit
# ---------------------------
model = st.session_state.get("model") or "gpt-4o-mini"  # from selectbox; keep safe default
temperature = st.session_state.get("temperature") or 0.3
prompt = st.chat_input("Ask something…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    context_block = ""
    refs_block = ""
    if st.session_state.get("use_rag", True):
        ctx = retrieve(prompt, k=5)
        if ctx:
            context_block = format_context(ctx)
            refs_block = build_apa_sources_note(ctx)

    sys_msg = st.session_state.messages[0]
    convo_msgs = [{"role": "system", "content": sys_msg["content"]}]
    if context_block:
        convo_msgs.append({
            "role": "system",
            "content": f"Use the following CONTEXT if helpful. If irrelevant, ignore.\n\nCONTEXT START\n{context_block}\nCONTEXT END"
        })
    for m in st.session_state.messages[1:]:
        if m["role"] in ("user", "assistant"):
            convo_msgs.append(m)

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("Thinking…"):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=convo_msgs,
                    temperature=temperature,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Error from model: {e}"

            st.markdown(answer + (("\n\n" + refs_block) if refs_block else ""))
    st.session_state.messages.append({"role": "assistant", "content": answer + (("\n\n" + refs_block) if refs_block else "")})

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Tip: Rotate the passcode from Streamlit **Secrets** when needed. Documents you upload are stored in the app server's Chroma DB and never pushed to Git.")
