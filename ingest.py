# ingest.py
# ingest.py

import os
from dotenv import load_dotenv
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from utils.parser import parse_document  # Custom parser module

# Load environment variables
load_dotenv()

# Set paths
DATA_DIR = "data"
CHROMA_DIR = "embeddings"

# Configure embedding model
embedding_model = OpenAIEmbeddings()

# Configure chunker
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
                metadata=[{"source": filename}]
            )

            documents.extend(chunks)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    print(f"✅ Parsed {len(documents)} chunks. Now embedding and storing...")

    # Store in Chroma
    Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_DIR)
    print(f"✅ Ingestion complete. Embeddings stored in '{CHROMA_DIR}'.")

if __name__ == "__main__":
    ingest()
