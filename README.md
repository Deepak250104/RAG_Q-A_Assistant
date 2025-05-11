# RAG-Powered Agentic Q\&A Assistant

## Overview

The **RAG-Powered Agentic Q\&A Assistant** is a machine learning-based assistant that uses Retrieval-Augmented Generation (RAG) to answer queries by retrieving relevant document chunks from a pre-built vector store and then generating responses using a language model.

This project leverages a combination of FAISS for efficient similarity search and Hugging Face embeddings for vector representations of text. It uses the LangChain framework to manage document ingestion, vector store indexing, and agent orchestration.

## Features

* **Document Ingestion**: Load and index documents into a vector store.
* **Vector Store**: Efficient storage of document embeddings for fast retrieval.
* **Query Handling**: Process user queries by retrieving the most relevant document chunks and generating responses.
* **Customization**: Easily replace the backend model or vector store for different applications.

## Requirements

Before running the project, ensure you have the following installed:

* Python 3.8 or higher
* Install dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

### Key Dependencies:

* **LangChain**: Framework to manage document load and agent orchestration.
* **FAISS**: For efficient vector search.
* **Hugging Face Transformers**: For embedding and language model support.
* **Pickle**: For storing and loading vector store metadata.
* **NumPy**: For numerical operations.

## File Structure

```plaintext
RAG-POWERED_AGENTIC_Q-A_ASSISTANT/
├── app/
│   └── backend/
│       ├── __pycache__/
│       ├── __init__.py
│       ├── agent.py            # Logic for agent that handles queries.
│       ├── ingest.py           # Document ingestion and vector store indexing.
│       ├── llm.py              # Logic to interact with language models.
│       ├── retriever.py        # Functions for retrieving relevant document chunks.
│       ├── tools.py            # Utility functions for the application.
│       └── utils.py            # Additional helper functions.
│   └── interface/
│       ├── __init__.py
│       └── ui.py               # User interface code.
├── data/
│   ├── documents/              # Text files for ingestion and indexing.
│   │   ├── ai_ethics.txt
│   │   ├── machine_learning.txt
│   │   ├── nlp_basics.txt
│   │   └── transformers.txt
│   └── processed/              # Processed data for the vector store.
│       └── vectorstore/
│           ├── index.faiss     # FAISS index file storing vector embeddings.
│           └── index.pkl       # Pickle file storing metadata for the vector store.
├── vectorstore/                # Vector store folder containing FAISS index.
│   ├── __pycache__/            
│   ├── __init__.py
│   ├── index.faiss             # FAISS index file for search.
│   ├── index.pkl               # Pickle file with metadata for the vector store.
│   └── index.py                # Code for vector store manipulation.
├── .env                         # Environment variables for configuration (e.g., API keys).
├── LICENSE                      # License file for the project.
├── MAIN.py                      # Main entry point for the application.
├── README.md                    # Project documentation (this file).
└── requirements.txt             # Python dependencies for the project.
```

### Explanation of Key Folders and Files:

* **`app/backend/`**: Contains all the backend logic of the project, including document ingestion, vector store creation, query handling, and agent orchestration.

  * **`ingest.py`**: Handles the ingestion of documents, vectorization of text, and storage in the vector store.
  * **`retriever.py`**: Contains functions to retrieve relevant document chunks from the vector store using FAISS.
  * **`llm.py`**: Manages interactions with the language model for generating responses based on retrieved document chunks.
  * **`tools.py`**: Contains utility functions such as formatting and processing data.
  * **`agent.py`**: Implements the core agent functionality, such as query processing and orchestrating responses from the model.
  * **`utils.py`**: Contains various helper functions used across the backend files.

* **`data/`**: Contains the raw documents that will be indexed as well as the processed vector store.

  * **`documents/`**: Holds the raw text files (`.txt`) to be indexed, such as AI ethics, machine learning topics, etc.
  * **`processed/`**: Contains the vector store used by the application to store embeddings and metadata.
  * **`vectorstore/`**: Holds the FAISS index (`index.faiss`) and metadata (`index.pkl`) files created after indexing documents.

* **`vectorstore/`**: This is the folder that contains the actual vector store for the application. It holds both the FAISS index and the associated pickle file with metadata.

* **`MAIN.py`**: The main entry point of the application. It coordinates document ingestion, query processing, and interaction with the vector store.

* **`requirements.txt`**: A file that lists all the dependencies required to run the project.

* **`.env`**: Stores environment variables, including any sensitive data like API keys (this file is usually not checked into version control for security reasons).

## Usage

### Ingesting Documents

To index documents into the vector store, run the following command:

```bash
python app/backend/ingest.py
```

This will process all `.txt` files in the `data/documents/` folder, convert them into embeddings, and store them in the `data/processed/vectorstore/` directory.

### Query Processing

To process a query using the vector store, run:

```bash
python main.py
```

This will start the application and process a sample query. The query will be matched with relevant document chunks, and a response will be generated based on the retrieved chunks.

### Testing and Development

* Modify the documents in `data/documents/` to add new topics or modify the content.
* To change the language model or vector store configurations, modify the relevant settings in the `.env` file and `llm.py`.
* To integrate with different models or vector stores, make sure to update the code in `ingest.py`, `retriever.py`, and `llm.py`.

## Contributing

Contributions are welcome! If you'd like to add features or improve the project, feel free to fork the repository and submit a pull request.

### Steps to Contribute:

1. Fork the repository.
2. Clone your forked repository to your local machine.
3. Create a new branch for your feature or fix.
4. Implement your changes.
5. Run the tests to make sure everything works.
6. Submit a pull request with a description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

