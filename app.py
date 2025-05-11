import os
import sys

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the create_interface function from the simple_qa module
# Note: You'll need to save the final_simple_qa code as simple_qa.py in your backend directory
from app.backend.simple_qa import create_interface

# Create and launch the interface
if __name__ == "__main__":
    # Create the Gradio interface
    demo = create_interface()
    
    # Launch the interface
    demo.queue()
    demo.launch(share=False)