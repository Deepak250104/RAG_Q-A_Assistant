import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from vectorstore.index import save_vector_store  # Now an absolute import should work

# Load environment variables
load_dotenv()

# Get the vector store path
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/processed/vectorstore/")

def ingest_documents():
    """
    Function to ingest documents, process them, and save to the vector store.
    """
    # Load documents
    documents = []
    for filename in os.listdir("data/documents"):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("data/documents", filename))
            documents.extend(loader.load())

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use FAISS for vector storage
    vectordb = FAISS.from_documents(documents, embeddings)

    # Save the vector store
    save_vector_store(vectordb, {"embeddings": embeddings}, VECTOR_DB_PATH)

    print(f"Ingested and indexed {len(documents)} document chunks.")

if __name__ == "__main__":
    ingest_documents()