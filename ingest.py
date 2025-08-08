# ingest.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from utils.parser import parse_document  # custom parser
from dotenv import load_dotenv
from tqdm import tqdm

# Load your OpenAI key
load_dotenv()
embedding_model = OpenAIEmbeddings()

# Paths
DATA_DIR = "data"
CHROMA_DIR = "embeddings"

# Chunker settings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def ingest():
    documents = []

    for filename in tqdm(os.listdir(DATA_DIR), desc="Processing documents"):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            content = parse_document(filepath)  # use your custom parser
            chunks = text_splitter.create_documents([content], metadata=[{"source": filename}])
            documents.extend(chunks)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    print(f"✅ Parsed {len(documents)} chunks. Now embedding and storing...")

    # Store in ChromaDB
    Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_DIR)
    print(f"✅ All chunks embedded and saved to '{CHROMA_DIR}'")

if __name__ == "__main__":
    ingest()
