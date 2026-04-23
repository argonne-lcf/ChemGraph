import math
import numexpr

from langchain_core.tools import Tool
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.types import interrupt


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


@tool
def ask_human(question: str) -> str:
    """Ask the human user for clarification, confirmation, or additional details.

    Use this tool when:
    - Required inputs are missing or ambiguous (e.g., molecule name, calculator
      type, temperature, pressure, or simulation method).
    - You need confirmation before running a computationally expensive simulation
      (e.g., geometry optimization, vibrational analysis, thermochemistry).
    - A previous tool call failed and you need the user to decide how to proceed
      (e.g., retry with different parameters, skip the step, or abort).

    The graph execution will pause until the human responds. The human's
    answer is returned as a string.

    Parameters
    ----------
    question : str
        The question or request to present to the human user.

    Returns
    -------
    str
        The human's response.
    """
    response = interrupt({"question": question})
    if isinstance(response, dict):
        return response.get("answer", response.get("response", str(response)))
    return str(response)


python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
