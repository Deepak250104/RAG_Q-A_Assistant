import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# Get the Hugging Face API token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_ID = "google/flan-t5-large"  

def run_agent(query, vectordb):
    # Initialize the LLM with correct parameter structure
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text2text-generation",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        model_kwargs={"timeout": 30}  # Moved from client_kwargs to model_kwargs
    )
    
    # Set up retriever tool
    retriever = vectordb.as_retriever()
    tools = [
        Tool(
            name="Retriever",
            func=retriever.get_relevant_documents,
            description="Retrieve relevant information from the vector store"
        )
    ]
    
    # Use specific AgentType enum instead of string
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    
    # Error handling for agent execution
    try:
        answer = agent.invoke({"input": query})
        return answer.get("output", "No response generated")
    except Exception as e:
        return f"An error occurred: {str(e)}"