# app.py
# Streamlit + Passcode Gate + RAG (Chroma) + Chat with OpenAI
# ---------------------------------------------------------------
# - Layout matches the "old" format: RAG toggle + Upload + Model/Temp in sidebar
# - Uses OpenAI embeddings (default: text-embedding-3-small, 1536-dim)
# - Avatars: assistant = kelp icon, user = test-tube icon
# - Metadata-aware retrieval: if the user hints an author/title (e.g., "Jose"),
#   we first filter by metadata (paper_title/authors), then do semantic ranking.
# ---------------------------------------------------------------

import os
import io
import re
import time
import json
import base64
import unicodedata
from typing import List, Tuple, Optional, Dict, Any
import reportlab
import streamlit as st
from pathlib import Path

# ---------- Page setup early (favicon/title on auth screen too) ----------
ROOT = Path(__file__).parent.resolve()

def _first_existing_local(*names: str) -> Optional[Path]:
    for n in names:
        if not n:
            continue
        p = (ROOT / n).resolve()
        if p.exists():
            return p
    return None

LOGO_FILE = _first_existing_local("logo_icon.png","logo_icon.jpg","kelp_ark_logo.png","kelp_ark_logo.jpg")
ASSISTANT_ICON_FILE = _first_existing_local("icon_kelp.png","icon_kelp.jpg","kelp_icon.png","kelp_icon.jpg","model_avatar.png")
USER_ICON_FILE      = _first_existing_local("test_tube.png","test_tube.jpg","icon_test_tube.png","icon_test_tube.jpg","user_model.png")

st.set_page_config(page_title="KelpGPT", page_icon=(str(LOGO_FILE) if LOGO_FILE else "🪸"), layout="wide")

# --- Local dev only: load .env if present ---
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
if "use_rag" not in st.session_state: st.session_state.use_rag = True
if "model" not in st.session_state: st.session_state.model = "gpt-4o-mini"
if "temperature" not in st.session_state: st.session_state.temperature = 0.3

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

# ---------- Chroma (modern API, matches ingest.py) ----------
import chromadb
from chromadb.utils import embedding_functions

@st.cache_resource(show_spinner=False)
def _get_chroma():
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )
    cli = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = cli.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"embedding_model": EMBED_MODEL},
        embedding_function=ef
    )
    return cli, col

def _get_collection():
    try:
        _, collection = _get_chroma()
        return collection
    except Exception as e:
        st.warning(f"RAG not available: {e}")
        return None

# ---------- PDF helpers ----------
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
        m = re.search(r"(19|20)\d{2}", y)
        if m: year = m.group(0)
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

# ---------- APA refs ----------
def _build_apa_line(meta: Dict[str, Any], fallback_filename: str) -> str:
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

# ---------- Metadata-aware retrieval helpers ----------
def _norm(s: Optional[str]) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return s.lower().strip()

def _extract_paper_hints(user_text: str) -> List[str]:
    t = _norm(user_text)
    hints: List[str] = []
    # quoted titles
    for q in re.findall(r"['\"]([^'\"]{3,80})['\"]", user_text):
        hints.append(_norm(q))
    tokens = re.findall(r"[a-zA-Z]{3,}", t)
    stop = {"paper","papers","about","the","and","from","study","article","report","figure","figures","seaweed","kelp","ulva","growth","what","know","joses"}
    for tok in tokens:
        if tok not in stop and 3 <= len(tok) <= 18:
            hints.append(tok)
    seen = set(); out=[]
    for h in hints:
        if h not in seen:
            seen.add(h); out.append(h)
    return out[:3]

def _candidate_doc_ids_by_metadata(collection, hints: List[str], limit_docs: int = 30) -> List[str]:
    got = collection.get(include=["metadatas"], limit=5000)
    metas = (got or {}).get("metadatas") or []
    doc_ids = []
    seen = set()
    for m in metas:
        title = _norm(m.get("paper_title") or m.get("title"))
        authors = _norm(m.get("authors"))
        doc_id = m.get("doc_id")
        if not doc_id:
            continue
        hay = f"{title} || {authors}"
        if any(h in hay for h in hints):
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
                if len(doc_ids) >= limit_docs:
                    break
    return doc_ids

# ---------- Sidebar (old layout) ----------
with st.sidebar:
    # One-time init for conversations (keeps your existing messages as the first chat)
    if "convos" not in st.session_state:
        st.session_state.convos = [{
            "id": int(time.time()*1000),
            "title": "New chat",
            "created": time.time(),
            "messages": st.session_state.get("messages", [])
        }]
        st.session_state.active_convo = 0

    # Convenience handles
    convos = st.session_state.convos
    active_idx = st.session_state.get("active_convo", 0)

    # Logo
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

    st.toggle(
        "Use document retrieval",
        value=st.session_state.use_rag,
        key="use_rag",
        help="Turn off to chat without pulling from your PDFs."
    )

    st.markdown("---")
    st.subheader("Conversations")

    # Selector
    labels = [(c["title"] if c["title"].strip() else "New chat")[:42] + ("" if len((c["title"] or "")) <= 42 else "…")
              for c in convos]
    selected = st.radio(
        label="",
        options=list(range(len(convos))),
        index=min(active_idx, len(convos)-1),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
        key="chat_selector",
    )
    if selected != active_idx:
        st.session_state.active_convo = selected
        active_idx = selected

    # Actions
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("➕ New"):
            convos.insert(0, {
                "id": int(time.time()*1000),
                "title": "New chat",
                "created": time.time(),
                "messages": []
            })
            st.session_state.active_convo = 0
            st.rerun()
    with c2:
        if st.button("🗑️ Delete", disabled=(len(convos) <= 1)):
            convos.pop(active_idx)
            st.session_state.active_convo = 0
            st.rerun()
    with c3:
        # Export current convo as PDF (fallback to JSON if reportlab not available)
        export_chat = convos[active_idx]
        pdf_bytes = None
        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            from reportlab.pdfbase import pdfmetrics
            import io as _io

            def _wrap_text(text: str, max_width: float, font_name: str, font_size: int) -> list:
                words = (text or "").split()
                if not words:
                    return [""]
                lines, cur = [], ""
                for w in words:
                    test = (cur + " " + w).strip()
                    if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
                        cur = test
                    else:
                        lines.append(cur)
                        cur = w
                if cur:
                    lines.append(cur)
                return lines

            buf = _io.BytesIO()
            c = canvas.Canvas(buf, pagesize=LETTER)
            width, height = LETTER

            left = 0.75 * inch
            right = 0.75 * inch
            top = 0.75 * inch
            bottom = 0.75 * inch
            y = height - top

            # Title
            title = (export_chat.get("title") or "KelpGPT Chat")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left, y, title)
            y -= 18

            c.setFont("Helvetica", 9)
            c.drawString(left, y, f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 12
            c.drawString(left, y, f"Chat ID: {export_chat.get('id')}")
            y -= 18

            # Horizontal rule
            c.setLineWidth(0.5)
            c.line(left, y, width - right, y)
            y -= 12

            # Messages
            for msg in export_chat.get("messages", []):
                role = msg.get("role", "").capitalize() or "Message"
                content = msg.get("content", "") or ""
                # Role header
                c.setFont("Helvetica-Bold", 11)
                role_line = f"{role}"
                role_h = 14
                if y - role_h < bottom:
                    c.showPage()
                    y = height - top
                c.drawString(left, y, role_line)
                y -= role_h

                # Content
                c.setFont("Helvetica", 10)
                max_w = (width - left - right)
                for line in _wrap_text(content, max_w, "Helvetica", 10):
                    line_h = 13
                    if y - line_h < bottom:
                        c.showPage()
                        y = height - top
                        c.setFont("Helvetica", 10)
                    c.drawString(left, y, line)
                    y -= line_h

                # Spacing after each message
                y -= 6
                if y < bottom:
                    c.showPage()
                    y = height - top

            c.save()
            pdf_bytes = buf.getvalue()
            buf.close()
        except Exception:
            pdf_bytes = None

        if pdf_bytes:
            st.download_button(
                "⬇️ Export PDF",
                data=pdf_bytes,
                file_name=f"kelpgpt_chat_{export_chat['id']}.pdf",
                mime="application/pdf"
            )
        else:
            # Fallback: JSON if PDF generation isn't available
            export_payload = json.dumps(export_chat, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Export (JSON)",
                data=export_payload,
                file_name=f"kelpgpt_chat_{export_chat['id']}.json",
                mime="application/json"
            )

    # Rename
    new_title = st.text_input(
        "Rename",
        value=convos[active_idx]["title"],
        max_chars=100,
        help="Rename this conversation."
    )
    if new_title != convos[active_idx]["title"]:
        convos[active_idx]["title"] = new_title

    st.markdown("---")

    # IMPORTANT: bind the selected conversation's messages to the app's message list
    st.session_state.messages = convos[active_idx]["messages"]

    # Dummy vars so any downstream ingest block won't break (even if you remove it later)
    up_files = None
    meta_sidecar = None
    ingest_click = False



# ---------- Main header ----------
st.markdown(
    """
    <div style="font-size:25px; font-weight:500; color:#cccccc; margin-bottom:4px;">
        KelpGPT – Internal research assistant
    </div>
    """,
    unsafe_allow_html=True
)

# Right-aligned logo
col_left, col_right = st.columns([1, 0.12])
with col_left:
    st.markdown(
        """
        <div style="font-size:35px; font-weight:600; line-height:1.1;">
            I'm KARA, how can I help you?
        </div>
        <div style="margin-top:6px; font-size:25px; font-weight:500; color:#aaaaaa;">
            KelpArk Research Assistant
        </div>
        """,
        unsafe_allow_html=True
    )
with col_right:
    if LOGO_FILE:
        st.image(str(LOGO_FILE), width=64)


# ---------- RAG status banner ----------
collection_for_status = _get_collection()
if collection_for_status is not None:
    try:
        st.info(f"Internal database is online.")
    except Exception:
        st.info(f"RAG online. Collection '{TEXT_COLLECTION}' is available.")
else:
    st.warning("RAG not available.")

st.markdown("---")



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

    # 1) Try metadata-guided narrowing (author/title hints)
    hints = _extract_paper_hints(query)
    docs: List[str] = []
    metas: List[dict] = []

    if hints:
        try:
            doc_ids = _candidate_doc_ids_by_metadata(collection, hints)
            if doc_ids:
                res = collection.query(
                    query_texts=[query],
                    n_results=k,
                    where={"doc_id": {"$in": doc_ids}},
                )
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
        except Exception:
            pass  # fall back

    # 2) Normal semantic retrieval if nothing matched
    if not docs:
        res = collection.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

    return list(zip(docs, metas))

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

# ---------- Chat submit ----------
prompt = st.chat_input("Ask something…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    ctx: List[Tuple[str, dict]] = []
    refs_block = ""
    if st.session_state.use_rag:
        ctx = retrieve(prompt, k=6)
        refs_block = build_apa_sources_note(ctx)

    context_block = _build_context_block(ctx) if ctx else ""

    # Build conversation payload
   # Build conversation payload
    convo_msgs = []
    convo_msgs.append({
        "role": "system",
        "content": (
            "You are KelpGPT, a marine science research assistant for Kelp Ark. "
            "Prefer information from the provided context. If the user asks about a specific paper "
            "(e.g., 'Jose's paper'), answer primarily from chunks whose metadata (paper_title/authors) match. "
            "If context is missing, say so briefly and proceed with best knowledge."
            "Include in-text citations for any data or reccomendations."
        )
    })


    if context_block:
        convo_msgs.append({
            "role": "system",
            "content": f"Context (top-k retrieved, cite by paper title in a References section):\n\n{context_block}"
        })

    # include prior turns for continuity
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant"):
            convo_msgs.append(m)

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("Thinking…"):
            try:
                resp = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=convo_msgs,
                    temperature=st.session_state.temperature,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Error from model: {e}"

            st.markdown(answer + (("\n\n" + refs_block) if refs_block else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer + (("\n\n" + refs_block) if refs_block else "")})

# ---------- Footer ----------
st.markdown("---")
st.caption("Tip: Rotate the passcode in Streamlit **Secrets** when needed. PDFs are stored in the app server's Chroma DB and never pushed to Git.")
