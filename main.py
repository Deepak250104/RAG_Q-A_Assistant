import os
from dotenv import load_dotenv
from app.backend.ingest import ingest_documents
from vectorstore.index import load_vector_store

# Load environment variables
load_dotenv()

# Get the vector store path
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/processed/vectorstore/")

def main():
    """
    Main function to process the documents and load the vector store (for potential checks).
    The Gradio app in app.py will handle the querying and response generation.
    """
    # Ingest documents and save them to the vector store
    print("Starting document ingestion...")
    ingest_documents()
    print("Document ingestion complete.")

    # Load the vector store (optional check)
    try:
        print(f"Checking vector store at {VECTOR_DB_PATH}...")
        vectordb, _ = load_vector_store(VECTOR_DB_PATH)
        print("Vector store loaded successfully.")
    except FileNotFoundError:
        print("Vector store not found. Ingestion might have failed or not been run.")
    except Exception as e:
        print(f"An error occurred while loading the vector store: {e}")

if __name__ == "__main__":
    main()