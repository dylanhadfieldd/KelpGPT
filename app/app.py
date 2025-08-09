# app.py
# Streamlit + Passcode Gate + Optional RAG (Chroma) + Chat with OpenAI
# ---------------------------------------------------------------
# - Key is stored in Streamlit Secrets (cloud) or .env (local)
# - Teammates use a 4-digit passcode; your key is never revealed
# - Optional: upload PDFs -> build local vector store (Chroma) for retrieval
# ---------------------------------------------------------------

import os
import time
from typing import List, Tuple, Optional
import streamlit as st

# --- logo path ---
LOGO_PATH = os.getenv("LOGO_PATH", "kelp_ark_logo.jpg")

# Configure page ASAP so favicon/title appear on auth screen too
st.set_page_config(page_title="KelpGPT", page_icon=LOGO_PATH, layout="wide")

# --- Local dev only: load .env if present (ignored in git) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------
# Secrets / Config helpers
# ---------------------------
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # Prefer Streamlit secrets (cloud), fallback to env (local)
    try:
        value = st.secrets.get(name)  # None if missing
        if value is None:
            return os.getenv(name, default)
        return value
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
# Auth Gate (4-digit passcode)
# ---------------------------
ATTEMPT_LIMIT = 5
LOCKOUT_SECS  = 120
SESSION_TTL   = 60 * 60  # 1 hour

if "authed" not in st.session_state:
    st.session_state.authed = False
if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "lockout_until" not in st.session_state:
    st.session_state.lockout_until = 0.0
if "auth_time" not in st.session_state:
    st.session_state.auth_time = 0.0

def require_auth() -> bool:
    now = time.time()

    # Expire session after TTL
    if st.session_state.authed and (now - st.session_state.auth_time > SESSION_TTL):
        st.session_state.authed = False

    if st.session_state.authed:
        return True

    # Lockout handling
    if now < st.session_state.lockout_until:
        remaining = int(st.session_state.lockout_until - now)
        st.title("🔒 Team Access")
        st.warning(f"Too many attempts. Try again in {remaining} seconds.")
        st.stop()

    st.title("🔒 Team Access")
    st.caption("Enter the 4‑digit passcode to use KelpGPT.")

    col1, col2 = st.columns([3,1])
    with col1:
        code = st.text_input("Passcode", type="password", max_chars=4,
                             help="Ask Dylan for the current code.")
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

# Gate the rest of the app
if not require_auth():
    st.stop()

# ---------------------------
# OpenAI client
# ---------------------------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Optional: Chroma RAG setup
# ---------------------------
@st.cache_resource(show_spinner=False)
def _get_chroma():
    import chromadb
    from chromadb.utils import embedding_functions
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=TEXT_COLLECTION,
        embedding_function=ef
    )
    return client, collection

def _chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    # Simple word-boundary chunking to stay under token limits
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end - overlap > start else end
    return chunks

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    import io
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

# ---------------------------
# UI Layout
# ---------------------------
with st.sidebar:
    # Try Streamlit's built-in logo helper (v1.31+), else fallback to image
    try:
        st.logo(LOGO_PATH)
    except Exception:
        st.image(LOGO_PATH, use_container_width=True)
    st.header("KelpGPT")
    st.caption("Internal research assistant")

    use_rag = st.toggle("Use document retrieval (RAG)", value=True,
                        help="When enabled, the app searches your uploaded/ingested PDFs for relevant context.")

    st.markdown("---")
    st.subheader("📥 Upload PDFs (optional)")
    up_files = st.file_uploader(
        "Add PDFs to the local index (Chroma). These are stored on the app server only.",
        type=["pdf"], accept_multiple_files=True
    )
    if st.button("Ingest PDFs"):
        if not up_files:
            st.warning("No PDFs selected.")
        else:
            with st.spinner("Ingesting…"):
                _, collection = _get_chroma()
                add_count = 0
                for f in up_files:
                    raw = f.read()
                    text = _extract_text_from_pdf(raw)
                    if not text.strip():
                        continue
                    chunks = _chunk_text(text)
                    # Use content-addressable IDs so re-ingesting is idempotent-ish
                    import hashlib
                    base = hashlib.sha1(raw).hexdigest()
                    ids = [f"{base}-{i}" for i in range(len(chunks))]
                    metadatas = [{"filename": f.name, "chunk": i} for i in range(len(chunks))]
                    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
                    add_count += len(chunks)
                st.success(f"Ingested {add_count} chunks.")

    st.markdown("---")
    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.3, 0.1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are KelpGPT, a precise, helpful marine science research assistant. Cite sources if provided in context."}
    ]

# --- Main header row (text left, logo right) ---
left, right = st.columns([5, 1])
with left:
    st.markdown("## I'm KARA, how can I help you?")
    st.caption("KelpArk Research Assistant")
with right:
    st.image(LOGO_PATH, use_container_width=True)

# Render prior messages (user/assistant only) with avatars
for m in st.session_state.messages:
    if m["role"] in ("user", "assistant"):
        avatar = "🙂" if m["role"] == "user" else LOGO_PATH
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

# ---------------------------
# Retrieval helper
# ---------------------------
def retrieve(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    """Return list of (doc, metadata)"""
    try:
        _, collection = _get_chroma()
    except Exception as e:
        st.warning(f"RAG not available: {e}")
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
        src = meta.get("filename", "doc")
        chunk = meta.get("chunk", i)
        lines.append(f"[{i}] ({src} • chunk {chunk})\n{doc}")
    return "\n\n".join(lines)

# ---------------------------
# Chat submit
# ---------------------------
prompt = st.chat_input("Ask something…")
if prompt:
    # add user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)

    # retrieve context if enabled
    context_block = ""
    sources_note = ""
    if use_rag:
        ctx = retrieve(prompt, k=5)
        if ctx:
            context_block = format_context(ctx)
            sources = [f"{m.get('filename','doc')}:chunk{m.get('chunk','?')}" for _, m in ctx]
            sources_note = "\n\n(Sources: " + ", ".join(sources) + ")"

    # Build messages for Chat Completions
    sys_msg = st.session_state.messages[0]  # system
    convo_msgs = [{"role": "system", "content": sys_msg["content"]}]
    if context_block:
        convo_msgs.append({
            "role": "system",
            "content": f"Use the following CONTEXT if helpful. If irrelevant, ignore.\n\nCONTEXT START\n{context_block}\nCONTEXT END"
        })
    # append conversation so far (excluding first system, which we manually added)
    for m in st.session_state.messages[1:]:
        if m["role"] in ("user", "assistant"):
            convo_msgs.append(m)

    # Call OpenAI
    with st.chat_message("assistant", avatar=LOGO_PATH):
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

            st.markdown(answer + (sources_note if context_block else ""))
    # add assistant msg
    st.session_state.messages.append(
        {"role": "assistant", "content": answer + (sources_note if context_block else "")}
    )

# ---------------------------
# Footer / tips
# ---------------------------
st.markdown("---")
st.caption(
    "Tip: Rotate the passcode from Streamlit **Secrets** when needed. "
    "Documents you upload are stored in the app server's Chroma DB and never pushed to Git."
)
