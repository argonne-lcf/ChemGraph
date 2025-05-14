from langchain_core.tools import tool
import math
import numexpr
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely.

    This function provides a safe way to evaluate mathematical expressions
    using numexpr. It supports basic mathematical operations and common
    mathematical functions.

    Parameters
    ----------
    expression : str
        Mathematical expression to evaluate (e.g., "2 * pi + 5")

    Returns
    -------
    str
        String result or error message

    Notes
    -----
    Supported mathematical functions:
    - Basic operations: +, -, *, /, **
    - Trigonometric: sin, cos, tan
    - Other: sqrt, abs
    - Constants: pi, e
    """
    local_dict = {
        "pi": math.pi,
        "e": math.e,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "abs": abs,
    }

    try:
        cleaned_expression = expression.strip()
        if not cleaned_expression:
            return "Error: Empty expression"

        result = numexpr.evaluate(
            cleaned_expression,
            global_dict={},
            local_dict=local_dict,
        )

        if isinstance(result, (int, float)):
            return f"{float(result):.6f}".rstrip("0").rstrip(".")
        return str(result)

    except Exception as e:
        return f"Error evaluating expression: {e!s}"


python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
