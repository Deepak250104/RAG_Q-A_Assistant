"""
Text generation using Hugging Face Inference API (e.g., google/flan-t5-xl).
"""
import os
import requests

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xl"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

prompt_template = """
Use the following context to answer the user's question.

Context:
{context}

Question: {query}
Answer:
"""

def generate_answer(query: str, context: str) -> str:
    """
    Generates an answer using context and the query via Hugging Face Inference API.
    """
    prompt = prompt_template.format(context=context, query=query)
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
    try:
        return response.json()[0]['generated_text']
    except:
        return "Sorry, something went wrong with the language model."
