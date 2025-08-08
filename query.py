# query.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Paths
CHROMA_DIR = "embeddings"

# Initialize embedding model & vector store
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

# Create retriever and LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(temperature=0, model="gpt-4")

# Chain to combine retriever + LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

def main():
    print("📚 Ask your research assistant anything. Type 'exit' to quit.")
    while True:
        query = input("\n🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            result = qa_chain({"query": query})
            print(f"\n💡 Answer:\n{result['result']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
