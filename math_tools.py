# Define some functions for mathematical operations used by model
import sympy as sp



tools_info = [
    {
        "name": "evaluate_expression",
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate (as a string)."
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "solve_equation",
        "description": "Solve the given equation for the specified variable.",
        "parameters": {
            "type": "object",
            "properties": {
                "equation": {
                    "type": "string",
                    "description": "The equation to solve (as a string)."
                },
                "variable": {
                    "type": "string",
                    "description": "The variable to solve for (as a string)."
                }
            },
            "required": ["equation", "variable"]
        }
    },
    {
        "name": "calculate_limit",
        "description": "Calculate the limit of the given expression as the variable approaches a point.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate (as a string)."
                },
                "variable": {
                    "type": "string",
                    "description": "The variable in the expression (as a string)."
                },
                "point": {
                    "type": "string",
                    "description": "The point to approach (can be a number or 'oo' for infinity)."
                }
            },
            "required": ["expression", "variable", "point"]
        }
    },
    {
        "name": "calculate_derivative",
        "description": "Calculate the derivative of the given expression with respect to the specified variable.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to differentiate (as a string)."
                },
                "variable": {
                    "type": "string",
                    "description": "The variable to differentiate with respect to (as a string)."
                }
            },
            "required": ["expression", "variable"]
        }
    },
    {
        "name": "calculate_integral",
        "description": "Calculate the integral of the given expression with respect to the specified variable.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to integrate (as a string)."
                },
                "variable": {
                    "type": "string",
                    "description": "The variable to integrate with respect to (as a string)."
                },
                "lower_limit": {
                    "type": "string",
                    "description": "The lower limit of integration (optional)."
                },
                "upper_limit": {
                    "type": "string",
                    "description": "The upper limit of integration (optional)."
                }
            },
            "required": ["expression", "variable"]
        }
    },
    {
        "name": "calculate_matrix_trace",
        "description": "Calculate the trace of a given matrix.",
        "parameters": {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "string",
                    "description": "The matrix to calculate the trace of (as a string representation, e.g., '[[1, 2], [3, 4]]')."
                }
            },
            "required": ["matrix"]
        }
    },
    {
        "name": "calculate_matrix_determinant",
        "description": "Calculate the determinant of a given matrix.",
        "parameters": {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "string",
                    "description": "The matrix to calculate the determinant of (as a string representation, e.g., '[[1, 2], [3, 4]]')."
                }
            },
            "required": ["matrix"]
        }
    },
    {
        "name": "calculate_matrix_inverse",
        "description": "Calculate the inverse of a given matrix.",
        "parameters": {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "string",
                    "description": "The matrix to calculate the inverse of (as a string representation, e.g., '[[1, 2], [3, 4]]')."
                }
            },
            "required": ["matrix"]
        }
    }
]


def execute_tool_call_str(tool_call_str):
    """
        tool_call_str: A string representation of the tool call, e.g., evaluate_expression, 2+2
    """
    tool_call_parts = tool_call_str.split(", ")
    tool_name = tool_call_parts[0]
    arguments = {
        k: v for k, v in zip(tools_info[0]["parameters"]["properties"].keys(), tool_call_parts[1:])
    }
    return execute_tool_call(tool_name, arguments)


# ------------------------Tool execution function------------------------
def execute_tool_call(tool_name, arguments):
    """
    Execute a tool call based on the tool name and arguments.
    :param tool_name: The name of the tool to execute.
    :param arguments: The arguments for the tool (as a dictionary).
    :return: The result of the tool execution - {function_name: function_name, result: result} or an error message.
    """
    if tool_name == "evaluate_expression":
        res = evaluate_expression(arguments["expression"])
    elif tool_name == "solve_equation":
        res = solve_equation(arguments["equation"], arguments["variable"])
    elif tool_name == "calculate_limit":
        res = calculate_limit(arguments["expression"], arguments["variable"], arguments["point"])
    elif tool_name == "calculate_derivative":
        res = calculate_derivative(arguments["expression"], arguments["variable"])
    elif tool_name == "calculate_integral":
        lower_limit = arguments.get("lower_limit", None)
        upper_limit = arguments.get("upper_limit", None)
        res = calculate_integral(arguments["expression"], arguments["variable"], lower_limit, upper_limit)
    elif tool_name == "calculate_matrix_trace":
        res = calculate_matrix_trace(arguments["matrix"])
    elif tool_name == "calculate_matrix_determinant":
        res = calculate_matrix_determinant(arguments["matrix"])
    elif tool_name == "calculate_matrix_inverse":
        res = calculate_matrix_inverse(arguments["matrix"])
    else:
        return {
            "function": "execute_tool_call",
            "error": f"Unknown tool: {tool_name}"
        }
    
    if isinstance(res, str):
        return {
            "function": tool_name,
            "result": res
        }
    
    # If the result is not a string, it must be a dict with error message
    return res
    

# Error message handler decorator
def error_message_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Return function name and error message
            return {
                "function": func.__name__,
                "error": str(e)
            }
    return wrapper

# ------------------------Expression evaluation functions------------------------
@error_message_handler
def evaluate_expression(expression):
    """
    Evaluate a mathematical expression and return the result.
    :param expression: The expression to evaluate (as a string).
    :return: The evaluated result as a string or an error message.
    """
    # Parse the expression
    expr = sp.sympify(expression)
    # Evaluate the expression
    result = expr.evalf()
    return str(result)


# ------------------------Equation solving functions------------------------
@error_message_handler
def solve_equation(equation, variable):
    """
    Solve the given equation for the specified variable.
    :param equation: The equation to solve (as a string).
    :param variable: The variable to solve for (as a string).
    :return: The solution as a string or an error message.
    """
    # Parse the equation and variable
    eq = sp.sympify(equation)
    var = sp.symbols(variable)

    # Solve the equation
    solution = sp.solve(eq, var)
    return str(solution)


# ------------------------Limit calculation functions------------------------
@error_message_handler
def calculate_limit(expression, variable, point):
    """
    Calculate the limit of the given expression as the variable approaches a point.
    :param expression: The expression to evaluate (as a string).
    :param variable: The variable in the expression (as a string).
    :param point: The point to approach (can be a number or 'oo' for infinity).
    :return: The limit as a string or an error message.
    """
    # Parse the expression and variable
    expr = sp.sympify(expression)
    var = sp.symbols(variable)

    # Handle special case for infinity
    if point == '-oo' or point == '-∞' or '-inf' in point:
        limit_value = sp.limit(expr, var, -sp.oo)
    elif point == 'oo' or point == '∞' or '+inf' in point or 'inf' in point:
        limit_value = sp.limit(expr, var, sp.oo)
    else:
        limit_value = sp.limit(expr, var, float(point))

    return str(limit_value)


# ------------------------Derivative calculation functions------------------------
@error_message_handler
def calculate_derivative(expression, variable):
    """
    Calculate the derivative of the given expression with respect to the specified variable.
    :param expression: The expression to differentiate (as a string).
    :param variable: The variable to differentiate with respect to (as a string).
    :return: The derivative as a string or an error message.
    """
    # Parse the expression and variable
    expr = sp.sympify(expression)
    var = sp.symbols(variable)

    # Calculate the derivative
    derivative = sp.diff(expr, var)
    return str(derivative)




# ------------------------Integral calculation functions------------------------
@error_message_handler
def calculate_integral(expression, variable, lower_limit=None, upper_limit=None):
    """
    Calculate the integral of the given expression with respect to the specified variable.
    :param expression: The expression to integrate (as a string).
    :param variable: The variable to integrate with respect to (as a string).
    :param lower_limit: The lower limit of integration (optional, can be None).
    :param upper_limit: The upper limit of integration (optional, can be None).
    :return: The integral as a string or an error message.
    """
    # Parse the expression and variable
    expr = sp.sympify(expression)
    var = sp.symbols(variable)

    # Calculate the integral
    if lower_limit is not None and upper_limit is not None:
        integral = sp.integrate(expr, (var, float(lower_limit), float(upper_limit)))
    else:
        integral = sp.integrate(expr, var)

    return str(integral)



# -----------------------Matrix Parse functions------------------------
@error_message_handler
def parse_matrix(matrix_str):
    """
    Parse a string representation of a matrix into a sympy Matrix object.
    :param matrix_str: The string representation of the matrix. e.g., "[[1, 2], [3, 4]]"
    :return: A sympy Matrix object or an error message.
    """
    # Convert the string to a sympy Matrix
    matrix = sp.Matrix(eval(matrix_str))
    return matrix

# ------------------------Matrix Trace functions------------------------
@error_message_handler
def calculate_matrix_trace(matrix):
    """
    Calculate the trace of a given matrix.
    :param matrix: The matrix to calculate the trace of (as a string).
    :return: The trace of the matrix as a string or an error message.
    """
    # Parse the matrix
    mat = parse_matrix(matrix)

    # Calculate the trace
    trace_value = mat.trace()
    return str(trace_value)


# ------------------------Matrix Determinant functions------------------------
@error_message_handler
def calculate_matrix_determinant(matrix):
    """
    Calculate the determinant of a given matrix.
    :param matrix: The matrix to calculate the determinant of (as a string).
    :return: The determinant of the matrix as a string or an error message.
    """
    # Parse the matrix
    mat = parse_matrix(matrix)

    # Calculate the determinant
    determinant_value = mat.det()
    return str(determinant_value)



# ------------------------Matrix Inverse functions------------------------
@error_message_handler
def calculate_matrix_inverse(matrix):
    """
    Calculate the inverse of a given matrix.
    :param matrix: The matrix to calculate the inverse of (as a string).
    :return: The inverse of the matrix as a string or an error message.
    """
    # Parse the matrix
    mat = parse_matrix(matrix)

    if mat.det() == 0:
        return "Error: The matrix is singular and cannot be inverted."
    
    # Calculate the inverse
    inverse_matrix = mat.inv()
    return str(inverse_matrix)
