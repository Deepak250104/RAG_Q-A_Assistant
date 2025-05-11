"""
Utility functions for embeddings, chunk loading, and I/O.
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_PATH = "data/processed/docs.txt"

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list:
    """
    Returns vector embedding of input text using SentenceTransformer.
    """
    return model.encode(text).tolist()

def load_documents(path=CHUNK_PATH):
    """
    Loads processed document chunks from file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_embeddings(path: str):
    """
    Loads document embeddings from .npy file.
    """
    return np.load(path)