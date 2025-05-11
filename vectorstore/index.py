import faiss
import pickle
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # Or your chosen embedding

# Path constants for the vector store files (consider using os.path.join for robustness)
INDEX_PATH = os.path.join("data", "processed", "vectorstore", "index.faiss")
PICKLE_PATH = os.path.join("data", "processed", "vectorstore", "index.pkl")

def load_vector_store(vectorstore_path):
    """
    Load the Langchain FAISS vector store from the provided path.
    """
    faiss_index_path = os.path.join(vectorstore_path, "index.faiss")
    faiss_data_path = os.path.join(vectorstore_path, "index.pkl")

    if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_data_path):
        raise FileNotFoundError(f"Vector store files not found at {vectorstore_path}")

    try:
        # Load FAISS index
        index = faiss.read_index(faiss_index_path)

        # Load associated data (embeddings, docstore, index_to_docstore_id)
        with open(faiss_data_path, "rb") as f:
            index_data = pickle.load(f)
            embeddings = index_data["embeddings"]
            docstore = index_data["docstore"]
            index_to_docstore_id = index_data["index_to_docstore_id"]

        # Create Langchain FAISS vector store
        vectordb = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)

        print(f"Langchain FAISS vector store loaded from {vectorstore_path}")
        return vectordb, index_data
    except FileNotFoundError:
        print("Vector store not found. Please ingest documents first.")
        raise
    except Exception as e:
        print(f"Error loading Langchain FAISS vector store: {e}")
        raise

def save_vector_store(vectordb, index_data, vectorstore_path):
    """
    Save the Langchain FAISS vector store (FAISS index and associated data) to the provided path.
    """
    try:
        # Save FAISS index
        faiss.write_index(vectordb.index, os.path.join(vectorstore_path, "index.faiss"))

        # Save associated data (embeddings, docstore, index_to_docstore_id)
        index_data_to_save = {
            "embeddings": vectordb.embedding_function,
            "docstore": vectordb.docstore,
            "index_to_docstore_id": vectordb.index_to_docstore_id,
        }
        with open(os.path.join(vectorstore_path, "index.pkl"), "wb") as f:
            pickle.dump(index_data_to_save, f)

        print(f"Langchain FAISS vector store saved to {vectorstore_path}")
    except Exception as e:
        print(f"Error saving Langchain FAISS vector store: {e}")
        raise