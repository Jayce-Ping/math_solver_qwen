import json
from math_tools import tools_info

initial_system_prompt = """You are an expert in solving mathematical problems."""

initial_prompt="""
The image contains a {problem_tag} mathematical problem.
Please extract the problem information and solve it step-by-step.
For such {problem_tag} problem, provide your final answer {answer_format}.
Contain your final answer within <answer>...</answer>"""

def format_prompt(problem_tag):
    tag_to_en = {
        '选择题': "multiple-choice",
        '填空题': "fill-in-the-blank",
        '计算应用题': "computational"
    }
    tag_to_answer_format = {
        '选择题': "as one of the options (A, B, C, D, etc.)",
        '填空题': "as a numerical value or expression",
        '计算应用题': "as a numerical value or expression"
    }
    return initial_prompt.format(problem_tag=tag_to_en[problem_tag], answer_format=tag_to_answer_format[problem_tag])


def format_query(query):
    return initial_prompt + "\n" + "Query: \n" + query


system_prompt_with_tool_calls = f"""{initial_system_prompt}
You have access to the following tools:
{json.dumps(tools_info, indent=2)}

Please use <tool_call>tool_name, arg1, arg2, ...</tool_call> to call a tool.
For example, if you want to call the evaluate_expression tool with the expression '2 + 2', you would write: <tool_call>evaluate_expression, 2 + 2</tool_call>.
"""