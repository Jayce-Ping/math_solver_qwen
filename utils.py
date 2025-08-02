import os
import json
import re
import yaml
from latex2sympy2 import latex2sympy
from sympy import N


default_model_load_kwargs = {
    "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
    'gpu_memory_utilization': 0.85,
    'max_model_len': 32768,
    'dtype': 'bfloat16',
    'device_cnt': 1
}

default_inference_kwargs = {
    'max_tokens': 4096,
    'temperature': 0.1,
    'top_p': 0.9,
    'batch_size': 4,  # Parallel processing batch size
}

def load_config(config_path = 'config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def is_numerical(s):
    """Check if a string can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_numerical_answer(answer):
    """Check if the answer is a numerical value."""
    try:
        float(answer)
        return True
    except ValueError:
        return False
    
def is_choice_answer(answer, choice_range=range(ord('a'), ord('z') + 1)):
    """Check if the answer is a choice (e.g., A, B, C, D)."""

    lower_answer = answer.lower()
    return lower_answer in [chr(i) for i in choice_range]

def calculate_fraction(answer):
    """
    Calculate the numerical value of a fraction in LaTeX format.
    """
    fraction_match = re.search(r'\\frac{(.*?)}{(.*?)}', answer)
    if fraction_match:
        numerator = fraction_match.group(1)
        denominator = fraction_match.group(2)
        try:
            return float(numerator) / float(denominator)
        except ValueError:
            pass
    return None

def calculate_square_root(answer):
    """
    Calculate the numerical value of a square root in LaTeX format.
    """
    sqrt_match = re.search(r'\\sqrt{(.*?)}', answer)
    if sqrt_match:
        value = sqrt_match.group(1)
        try:
            return float(value) ** 0.5
        except ValueError:
            pass
    return None

def calculate_power(answer):
    """
    Calculate the numerical value of a power in LaTeX format.
    """
    power_match = re.search(r'(.+)\^\{(.*?)\}', answer)
    if power_match:
        base = float(power_match.group(1))
        exponent = float(power_match.group(2))
        return base ** exponent
    return None

def calculate_answer(answer):
    """
    Calculate the numerical value of an answer in LaTeX format.
    This function handles fractions, square roots, and powers.
    """
    if is_numerical(answer):
        return float(answer)

    # Try to use the lib first
    try:
        # Use latex2sympy to convert LaTeX to a sympy expression
        sympy_expr = latex2sympy(answer)
        # Calculate the numerical value
        numerical_value = N(sympy_expr)
        return float(numerical_value)
    except Exception as e:
        pass


    # Use custom calculations for common LaTeX patterns - should not happen since they should be easily handled by latex2sympy
    calculators = [
        calculate_fraction,
        calculate_square_root,
        calculate_power,
    ]
    for calculator in calculators:
        try:
            value = calculator(answer)
            if value is not None:
                return value
        except:
            pass
        
    return None

def format_answer(answer, tag=None):
    """
        Remove latex formatting and ensure the answer is in a valid format.
    """
    # Remove \\boxed{...} with ...
    answer = re.sub(r'\\boxed{(.*?)}', r'\1', answer)

    # Remove \boxed{...} with ...
    answer = re.sub('\boxed{(.*?)}', r'\1', answer) # sometimes the mode returns \boxed{...} instead of \\boxed{...}

    # Remove \text{...} with ...
    answer = re.sub(r'\\text{(.*?)}', r'\1', answer)


    # Remove \\(...\\) with ...
    answer = re.sub(r'\\\((.*?)\\\)', r'\1', answer)

    if tag in ['选择题']:
        # Find the first uppercase letter in the answer
        match = re.search(r'[A-Z]', answer)
        if match:
            answer = match.group(0)
        else:
            # If no uppercase letter is found, return the answer as A
            return 'A'

    # For '填空题', '计算应用题', or other numerical answers
    if is_numerical_answer(answer):
        return f"{float(answer):.10f}"

    # Try to calculate the answer if it contains LaTeX formatting
    calculated_value = calculate_answer(answer)
    if calculated_value is not None and is_numerical(calculated_value):
        return f"{calculated_value:.10f}" 

    return answer.strip()

def extract_steps_and_answer(response):
    """
    Extract thinking process and answer
    <think> thinking process </think>
    <answer> (?Final answer):...</answer> ("Final Answer" is optional)
    """
    # Match steps and answer using regex
    steps_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    steps = steps_match.group(1).strip() if steps_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    if "Final Answer:" in answer:
        answer = answer.replace("Final Answer:", "").strip()

    return steps, answer

def load_jsonl(input_file):
    """Load jsonl File"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data