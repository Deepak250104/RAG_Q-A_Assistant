import faiss
import pickle
import os

# Path constants for the vector store files
INDEX_PATH = "data/processed/vectorstore/index.faiss"
PICKLE_PATH = "data/processed/vectorstore/index.pkl"

def load_vector_store(vectorstore_path):
    """
    Load the vector store from the provided path.
    This function assumes the index is a FAISS index and the vector store is serialized in a pickle file.
    """
    try:
        # Load FAISS index
        index = faiss.read_index(os.path.join(vectorstore_path, "index.faiss"))
        
        # Load associated data (optional metadata)
        with open(os.path.join(vectorstore_path, "index.pkl"), "rb") as f:
            index_data = pickle.load(f)

        print(f"Vector store loaded from {vectorstore_path}")
        return index, index_data
    except FileNotFoundError:
        print("Vector store not found. Please ingest documents first.")
        raise

def save_vector_store(index, index_data, vectorstore_path):
    """
    Save the vector store (FAISS index and associated data) to the provided path.
    """
    try:
        # Save FAISS index
        faiss.write_index(index, os.path.join(vectorstore_path, "index.faiss"))
        
        # Save associated data (optional metadata)
        with open(os.path.join(vectorstore_path, "index.pkl"), "wb") as f:
            pickle.dump(index_data, f)

        print(f"Vector store saved to {vectorstore_path}")
    except Exception as e:
        print(f"Error saving vector store: {e}")
