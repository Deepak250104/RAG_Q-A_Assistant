import gradio as gr
from app.backend.agent import handle_query  # Import the query handling function from agent.py

def generate_response(query):
    """
    Takes a user query, processes it through the agent framework, and returns the response.
    """
    response = handle_query(query)  # Uses the agent framework to process the query
    return response

iface = gr.Interface(
    fn=generate_response,               # Function to process user input
    inputs=gr.Textbox(label="Ask a Question"),  # Textbox for user input
    outputs=gr.Textbox(label="Response"),  # Output for the assistant's response
    live=True,                           # Update output live as user types
    title="RAG-Powered Q&A Assistant",   # Title for the interface
    description="A simple Q&A assistant using RAG for intelligent responses.",  # Description
)

if __name__ == "__main__":
    iface.launch()  # Launch the Gradio interface
