import os
from langchain.document_loaders import TextLoader
from vectorstore.index import save_vector_store
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the vector store path from environment variables (default to `data/processed/vectorstore/` if not set)
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/processed/vectorstore/")

def ingest_documents():
    """
    Function to ingest documents from the `data/documents/` directory, process them,
    and save them to the vector store.
    """
    # Load documents from the 'data/documents/' directory
    documents = []
    for filename in os.listdir("data/documents"):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("data/documents", filename))
            documents.extend(loader.load())

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use FAISS for vector storage
    vectordb = FAISS.from_documents(documents, embeddings)

    # Save the vector store to the specified path
    save_vector_store(vectordb.index, vectordb, VECTOR_DB_PATH)

    print(f"Ingested and indexed {len(documents)} document chunks.")
