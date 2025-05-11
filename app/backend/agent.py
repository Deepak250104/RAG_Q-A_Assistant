from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.tools import Tool  # Changed import
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

def run_agent(query, vectordb):
    """
    This function takes a query and retrieves relevant information from the vector store.
    It then uses the LangChain agent to process the query and return an answer.
    """
    # Step 1: Initialize the LLM and retriever
    llm = OpenAI(temperature=0)  # You can modify the temperature for the LLM
    retriever = vectordb.as_retriever()

    # Step 2: Initialize the agent with the retriever and LLM
    tools = [
        Tool(
            name="Retriever",
            func=retriever.retrieve,
            description="Retrieve relevant information from the vector store"
        )
    ]
    agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

    # Step 3: Use the agent to run the query and get an answer
    answer = agent.run(query)
    return answer