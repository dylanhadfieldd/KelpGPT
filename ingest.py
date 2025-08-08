# ingest.py

import os
from dotenv import load_dotenv
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from utils.parser import parse_document  # Custom file parser

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = "data"
EMBEDDINGS_DIR = "embeddings"

# Embedding model
embedding_model = OpenAIEmbeddings()

# Text splitter config
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def ingest():
    documents = []

    for filename in tqdm(os.listdir(DATA_DIR), desc="📄 Processing documents"):
        filepath = os.path.join(DATA_DIR, filename)

        try:
            content = parse_document(filepath)
            chunks = text_splitter.create_documents(
                [content],
                metadatas=[{"source": filename}]
            )
            documents.extend(chunks)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    print(f"✅ Parsed {len(documents)} chunks. Now embedding and storing...")

    # Use FAISS to store embeddings
    FAISS.from_documents(documents, embedding_model).save_local(EMBEDDINGS_DIR)
    print(f"✅ Ingestion complete. Embeddings stored in '{EMBEDDINGS_DIR}'.")

if __name__ == "__main__":
    ingest()
