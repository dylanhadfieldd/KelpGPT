#!/usr/bin/env python3
"""
Streamlit UI with:
- Q&A tab: Retrieval over Chroma -> answer with OpenAI (LangChain if available, otherwise a simple manual RAG).
- Figures tab: uses services.figure_index + our EmbeddingsService.
"""

from __future__ import annotations
import os
import streamlit as st

from config import (
    CHROMA_PERSIST_DIR_STR,
    OPENAI_API_KEY, OPENAI_EMBED_MODEL, LOCAL_EMBED_MODEL, EMBEDDING_PROVIDER,
)

st.set_page_config(page_title="KelpGPT", layout="wide")

# ---- Utilities ---------------------------------------------------------------
def make_embedder():
    # use your services/embeddings if present
    try:
        from services.embeddings import EmbeddingsService
        return EmbeddingsService()
    except Exception:
        pass

    # fallback local
    if EMBEDDING_PROVIDER.lower() == "local":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_EMBED_MODEL)
        class _Local:
            def embed_texts(self, texts):
                return model.encode(list(texts), normalize_embeddings=True).tolist()
        return _Local()

    # fallback OpenAI
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    class _OpenAI:
        def embed_texts(self, texts):
            resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=list(texts))
            return [d.embedding for d in resp.data]
    return _OpenAI()

def answer_with_openai(question: str, contexts: list[str]) -> str:
    """Lightweight composer when LangChain isn't available."""
    from openai import OpenAI
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set."
    client = OpenAI(api_key=OPENAI_API_KEY)

    sys = (
        "You are a helpful research assistant. "
        "Answer using ONLY the provided context. If unsure, say you don't know. "
        "Cite sources by file name and page when possible."
    )
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---- Tabs --------------------------------------------------------------------
tab = st.sidebar.radio("Sections", ["Q&A", "Figures"])

# ---------------------------- Q&A TAB ----------------------------------------
if tab == "Q&A":
    st.title("KelpGPT — Q&A")
    q = st.text_input("Ask a question about your library", placeholder="e.g., What affects Ulva growth rates?")
    k = st.slider("Top passages", 4, 20, 8)
    if st.button("Search", type="primary") and q:
        # Try LangChain retriever
        try:
            # LangChain path
            try:
                from langchain_openai import OpenAIEmbeddings
            except Exception:
                from langchain_community.embeddings import OpenAIEmbeddings  # legacy path

            from langchain_community.vectorstores import Chroma
            from langchain_core.prompts import PromptTemplate
            from langchain.chains import RetrievalQA
            from langchain_openai import ChatOpenAI

            if EMBEDDING_PROVIDER.lower() == "local":
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)
            else:
                embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)

            vectordb = Chroma(
                persist_directory=CHROMA_PERSIST_DIR_STR,
                embedding_function=embeddings,
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": k})

            prompt = PromptTemplate.from_template(
                "You are a helpful research assistant.\n"
                "Use the following context to answer the question. If unsure, say you don't know.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )

            llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0.2, api_key=OPENAI_API_KEY)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
            )
            res = qa.invoke({"query": q})

            st.subheader("Answer")
            st.write(res["result"])

            # Sources
            st.subheader("Sources")
            for d in res.get("source_documents", []):
                meta = d.metadata or {}
                st.write(f"- **{meta.get('source', meta.get('file_name',''))}**")
        except Exception as e:
            # Fallback: manual retrieval via chromadb + OpenAI
            import chromadb
            embedder = make_embedder()
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR_STR)
            col = client.get_or_create_collection(name="papers_text")
            qemb = embedder.embed_texts([q])
            result = col.query(query_embeddings=qemb, n_results=k)
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]

            answer = answer_with_openai(q, docs)
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for m in metas:
                st.write(f"- **{m.get('source', m.get('file_name',''))}**")

# ---------------------------- FIGURES TAB -------------------------------------
elif tab == "Figures":
    st.title("Figures")
    from services.figure_index import search_figures
    embedder = make_embedder()

    q = st.text_input("Search figures", placeholder="e.g., ulva growth patterns")
    k = st.slider("Results", 4, 36, 12, step=4)
    if st.button("Find Figures", type="primary") and q:
        qemb = embedder.embed_texts([q])
        res = search_figures(qemb, n_results=k)

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        if not ids:
            st.warning("No figures found.")
        else:
            cols = st.columns(3)
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                with cols[i % 3]:
                    st.image(meta.get("thumb_url") or meta.get("image_url"), use_column_width=True)
                    st.caption(doc or meta.get("caption_guess", ""))
                    st.write(f"**{meta.get('file_name','')}** — p.{meta.get('page','?')}")
                    if meta.get("image_url"):
                        st.link_button("Open image", meta["image_url"], use_container_width=True)
