"""
Simple tools for calculations and term definitions.
"""
def handle_calculation(query: str) -> str:
    """
    Evaluates simple math expressions from query.
    """
    try:
        expression = query.lower().replace('calculate', '').replace('compute', '').strip()
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def handle_definition(query: str) -> str:
    """
    Handles basic term definitions using static rules.
    """
    term = query.replace('define', '').replace('what is', '').strip().lower()
    dictionary = {
        "ai": "Artificial Intelligence is the simulation of human intelligence in machines.",
        "ml": "Machine Learning is a subset of AI focused on learning patterns from data.",
        "nlp": "Natural Language Processing deals with interaction between computers and human language."
    }
    return dictionary.get(term, f"No definition found for '{term}'.")