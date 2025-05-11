import os
from dotenv import load_dotenv
from app.backend.ingest import ingest_documents
from vectorstore.index import load_vector_store, save_vector_store
from app.backend.retriever import retrieve_relevant_chunks
from app.backend.llm import generate_answer

# Load environment variables
load_dotenv()

# Get the vector store path from environment variables (default to `data/processed/vectorstore/` if not set)
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/processed/vectorstore/")

def main():
    """
    Main function to process the documents, load the vector store, retrieve relevant chunks,
    and generate an answer for a sample query.
    """

    # Ingest documents and save them to the vector store
    print("Ingesting documents...")
    ingest_documents()

    # Load the vector store from the specified path
    try:
        print(f"Loading vector store from {VECTOR_DB_PATH}...")
        vectordb, vectordb_data = load_vector_store(VECTOR_DB_PATH)
    except FileNotFoundError:
        print("Vector store not found, please ingest documents first.")
        return

    # Sample query to retrieve relevant chunks from the vector store
    query = "What are the ethical principles of AI?"
    print(f"Processing query: {query}")
    retrieved_chunks = retrieve_relevant_chunks(query, vectordb)

    # Create context based on retrieved chunks or from a predefined source
    context = " ".join(retrieved_chunks)  # Combining the retrieved chunks as context

    # Generate the final answer from the query and the context
    print("Generating answer from retrieved chunks and context...")
    answer = generate_answer(query, context)  # Passing both query and context
    print(f"Answer: {answer}")

    # Optionally, save the updated vector store again after processing
    print("Saving vector store...")
    save_vector_store(vectordb, vectordb_data, VECTOR_DB_PATH)

if __name__ == "__main__":
    main()
