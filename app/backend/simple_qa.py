import os
import logging
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

def load_vectorstore(path="data/processed/vectorstore/"):
    """Load the vector store with safe deserialization"""
    print(f"Loading vector store from {path}...")
    if os.path.exists(path):
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()
        # Load existing vector store with safe deserialization
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

def simple_qa(query, vectordb):
    """Simple QA function using retrieval QA chain"""
    try:
        # Initialize Hugging Face LLM
        logger.info("Initializing Hugging Face LLM")
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-large",
            task="text2text-generation",
            temperature=0.1,
            max_new_tokens=512,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            model_kwargs={"timeout": 30}
        )
        
        # Create a retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        
        # Run the chain
        result = qa_chain.invoke({"query": query})
        
        # Format the response
        answer = result.get("result", "No answer found")
        
        # Add source documents if available
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append(f"- {doc.metadata.get('source', 'Unknown source')}")
        
        if sources:
            answer += "\n\nSources:\n" + "\n".join(sources)
            
        return answer
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}"

# Use this instead of run_agent in gradio.py
def run_agent(query, vectordb):
    """Compatibility wrapper for existing code"""
    return simple_qa(query, vectordb)

# Create a simple Gradio interface
def create_interface():
    import gradio as gr
    
    with gr.Blocks() as demo:
        gr.Markdown("# RAG-Powered Q&A Assistant")
        
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Ask a question")
        clear = gr.Button("Clear")

        def respond(message, history):
            # Add the user's message to history
            history.append({"role": "user", "content": message})
            
            # Get the response
            vectordb = load_vectorstore()
            if vectordb is None:
                response = "Error: Vector database could not be loaded."
            else:
                response = simple_qa(message, vectordb)
            
            # Add the bot's response to history
            history.append({"role": "assistant", "content": response})
            
            return "", history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo

# Launch the interface if run directly
if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch(share=False)