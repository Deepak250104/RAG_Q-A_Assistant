"""
Document retrieval using FAISS vector store and Hugging Face embeddings.
"""
import os
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# Path constants for the vector store
VECTOR_STORE_PATH = "data/processed/vectorstore/index.faiss"
VECTOR_STORE_PKL_PATH = "data/processed/vectorstore/index.pkl"

def load_vector_store():
    """
    Load the FAISS vector store from disk.
    """
    try:
        # Ensure the vector store exists at the correct path
        if os.path.exists(VECTOR_STORE_PATH):
            print(f"Loading vector store from {VECTOR_STORE_PATH}...")
            # Load the FAISS index from the specified path
            index = faiss.read_index(VECTOR_STORE_PATH)
            # Optionally load the associated metadata (if any) from the pkl file
            with open(VECTOR_STORE_PKL_PATH, 'rb') as f:
                metadata = pickle.load(f)
            print("Vector store loaded successfully.")
            return index, metadata
        else:
            print(f"Vector store file not found at {VECTOR_STORE_PATH}")
            return None, None
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None, None

def retrieve_relevant_chunks(query, vector_store):
    """
    Retrieve the most relevant document chunks for a given query from the vector store.
    """
    try:
        # Ensure the index is loaded before querying
        if vector_store is None:
            print("Vector store is not loaded.")
            return []
        retrieved_chunks = ["Placeholder chunk 1", "Placeholder chunk 2"]
        print(f"Retrieved chunks for query '{query}': {retrieved_chunks}")
        return retrieved_chunks

    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return []

