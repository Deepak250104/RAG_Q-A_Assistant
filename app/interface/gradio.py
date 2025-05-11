import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.backend.agent import run_agent

# Load or initialize vector store
def load_vectorstore(path="data/processed/vectorstore/"):
    print(f"Loading vector store from {path} in gradio.py...")
    if os.path.exists(path):
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()
        # Load existing vector store with allow_dangerous_deserialization=True
        # Note: Only do this if you trust the source of your vectorstore
        vectordb = FAISS.load_local(
            path, 
            embeddings, 
            allow_dangerous_deserialization=True  # Add this parameter
        )
        print("Langchain FAISS vector store loaded from", path)
        return vectordb
    else:
        print(f"Error: Vector store not found at {path}")
        return None

# Initialize vector DB
vectordb = load_vectorstore()

# Response generation function
def generate_response(query):
    if not query.strip():
        return "Please enter a question."
    
    if vectordb is None:
        return "Vector database not loaded. Please check the path."
    
    try:
        response = run_agent(query, vectordb)
        return response
    except Exception as e:
        return f"Error processing your query: {str(e)}"

# Gradio interface setup
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG-Powered Q&A Assistant")
        
        chatbot = gr.Chatbot(type="messages")  # Set type parameter explicitly
        msg = gr.Textbox(label="Ask a question")
        clear = gr.Button("Clear")

        def respond(message, history):
            # Add the user's message to history
            history.append({"role": "user", "content": message})
            
            # Get the response from our agent
            response = generate_response(message)
            
            # Add the bot's response to history
            history.append({"role": "assistant", "content": response})
            
            return "", history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch(share=False)