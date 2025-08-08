# 👇 SQLite compatibility patch (must be at the very top)
import os
import sys

os.environ["LANGCHAIN_USE_PYSQLITE3"] = "1"
import pysqlite3
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# --- Normal imports ---
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()

# Embedding + LLM setup
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="embeddings", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(temperature=0, model="gpt-4")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# --- UI ---
st.set_page_config(page_title="KelpGPT", layout="centered")
st.title("🧪 Kelp Research Assistant")

query = st.text_input("Ask a question about your data/documents:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        st.markdown(f"### 💡 Answer:\n{result['result']}")

        with st.expander("📄 Sources"):
            for doc in result["source_documents"]:
                st.markdown(f"• **{doc.metadata.get('source', 'Unknown')}**")
