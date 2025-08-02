initial_prompt="""You are an expert in solving mathematical problems. Please begin by extracting the math problem from the provided image, and then solve it.

Requirements:
	1. All mathematical formulas and symbols in your response must be written in LaTeX format.
	2. The final answer can only contain the final result option or number.

Strictly follow the format below in your output: <think>...</think><answer>...</answer>"""

def format_query(query):
    return initial_prompt + "\n" + "Query: \n" + query