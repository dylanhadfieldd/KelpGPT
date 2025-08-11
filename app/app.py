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
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from pathlib import Path
from PIL import Image

# ---------- Paths & Helpers ----------
ROOT = Path(__file__).parent.resolve()

def _first_existing_local(*names: str) -> Optional[Path]:
    for n in names:
        if not n:
            continue
        p = (ROOT / n).resolve()
        if p.exists() and p.is_file():
            return p
    return None

# Image files in SAME folder as app.py
LOGO_FILE = _first_existing_local("kelp_ark_logo.png","logo_icon.jpg","kelp_ark_logo.png","kelp_ark_logo.jpg")
ASSISTANT_ICON_FILE = _first_existing_local("model_avatar.png","icon_kelp.jpg","kelp_icon.png","kelp_icon.jpg")
USER_ICON_FILE      = _first_existing_local("user_model.png","test_tube.jpg","icon_test_tube.png","icon_test_tube.jpg")

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

# ---------- Auth Gate ----------
ATTEMPT_LIMIT = 5
LOCKOUT_SECS  = 120
SESSION_TTL   = 60 * 60  # 1 hour

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

# ---------- OpenAI client ----------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Chroma (matches ingest.py: OpenAI 1536-dim) ----------
@st.cache_resource(show_spinner=False)
def _get_chroma():
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",  # 1536-dim
    )
    cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = cli.get_or_create_collection(name=TEXT_COLLECTION, embedding_function=ef)
    return cli, col

# ---------- PDF helpers / APA ----------
def _chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if j - overlap > i else j
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

def _build_apa_citation(meta: Dict[str, Any], fallback_filename: str) -> str:
    authors = meta.get("authors")
    year    = meta.get("year")
    title   = meta.get("title")
    if not (authors or title):
        return fallback_filename
    parts = []
    if authors: parts.append(f"{authors}")
    if year:    parts.append(f"({year}).")
    if title:   parts.append(f"{title}.")
    return " ".join(parts).strip()

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# ---------- Sidebar ----------
with st.sidebar:
    if LOGO_FILE:
        st.logo(str(LOGO_FILE))
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

    if st.button("Ingest PDFs"):
        if not up_files:
            st.warning("No PDFs selected.")
        else:
            with st.spinner("Ingesting…"):
                _, collection = _get_chroma()
                add_count = 0
                for f in up_files:
                    raw = f.read()
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
                    base_meta = {
                        "source_filename": f.name,
                        "authors": core.get("authors"),
                        "year": core.get("year"),
                        "title": core.get("title"),
                    }

                    # Content-addressable IDs
                    import hashlib
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
    _, _col = _get_chroma()
    st.info(f"RAG online. Collection '{TEXT_COLLECTION}' contains ~{_col.count()} chunks.")
except Exception as e:
    st.warning(f"RAG not available: {e}")

# ---------- Avatars + history ----------
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

# ---------- Retrieval ----------
def _get_collection():
    try:
        _, collection = _get_chroma()
        return collection
    except Exception as e:
        st.warning(f"RAG not available: {e}")
        return None

def retrieve(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    collection = _get_collection()
    if collection is None:
        return []
    res = collection.query(query_texts=[query], n_results=k)
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

# ---------- Chat submit ----------
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
