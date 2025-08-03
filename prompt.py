import json
from math_tools import tools_info

initial_system_prompt = """You are an expert in solving mathematical problems. Please begin by extracting the math problem from the provided image, and then solve it. Contain the final result option or number within <answer>...</answer>."""

initial_prompt="""Solve the problem. Strictly follow the format below in your output: <think>...</think><answer>...</answer>"""

def format_query(query):
    return initial_prompt + "\n" + "Query: \n" + query


system_prompt_with_tool_calls = f"""{initial_system_prompt}
You have access to the following tools:
{json.dumps(tools_info, indent=2)}

Please use <tool>tool_name, arg1, arg2, ...</tool> to call a tool. For example, if you want to call the evaluate_expression tool with the expression '2 + 2', you would write: <tool>evaluate_expression, 2 + 2</tool>.
"""