import json
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def get_workflow_from_state(state) -> dict:
    """Convert a state object to a workflow dictionary.

    This function processes a state object containing AIMessages with tool calls
    and converts it into a standardized workflow dictionary format.

    Parameters
    ----------
    state : list
        List of messages, including AIMessages containing tool calls

    Returns
    -------
    dict
        A dictionary containing:
        - tool_calls: List of dictionaries mapping tool names to their arguments
        - result: The final result or answer from the workflow

    Notes
    -----
    The function processes:
    - AIMessages containing tool calls
    - The final message's content, which may be:
      - A JSON string with an 'answer' field
      - A JSON string with direct content
      - A plain string
      - Any other content type
    """
    workflow_dict = {"tool_calls": []}

    def recurse(obj):
        """Collect AI tool calls from nested workflow state objects.

        Parameters
        ----------
        obj : Any
            Dictionary, list, or scalar value from the serialized workflow.
        """
        if isinstance(obj, dict):
            # Extract tool_calls if it's an AI message
            if obj.get("type") == "ai":
                tool_calls = obj.get("tool_calls", [])
                for call in tool_calls:
                    name = call.get("name")
                    args = call.get("args", {})
                    workflow_dict["tool_calls"].append({name: args})
            # Recurse into all values
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(state)

    last_message = state["messages"][-1]

    content = last_message.get("content", {})

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass  # keep content as-is if it's not valid JSON

    # Extract result (just the value of the "answer" key if it exists)
    if isinstance(content, dict) and "answer" in content:
        workflow_dict["result"] = content["answer"]
    else:
        workflow_dict["result"] = content

    return workflow_dict
