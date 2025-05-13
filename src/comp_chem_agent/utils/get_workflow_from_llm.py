import json
from langchain.schema.messages import AIMessage
import logging
from comp_chem_agent.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def get_workflow_from_log(file_path: str) -> dict:
    """Convert a run_logs file to a workflow dictionary for evaluations.

    This function reads a JSON log file containing tool calls and their results,
    and converts it into a standardized workflow dictionary format.

    Parameters
    ----------
    file_path : str
        Path to the run logs file in JSON format

    Returns
    -------
    dict
        A dictionary containing:
        - tool_calls: List of tool call arguments
        - result: The final result or answer from the workflow

    Notes
    -----
    The function expects the log file to contain:
    - A 'state' list with tool calls and their arguments
    - A final message with either a JSON 'answer' field or direct content
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    # Extract tool names and arguments
    workflow_dict = {"tool_calls": []}
    for state in data.get("state", []):
        tool_calls = state.get("tool_calls", [])
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args")
            dat = {}
            dat[name] = args
            workflow_dict["tool_calls"].append(args)
    last_message = data.get("state", [])[-1]
    try:
        if "answer" in last_message["content"]:
            result_data = json.loads(last_message["content"])
            workflow_dict["result"] = result_data.get("answer")
    except Exception as e:
        result_data = last_message["content"]
        workflow_dict["result"] = result_data
        logging.debug(f"Exception thrown while parsing result: {e}")

    return workflow_dict


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
    for msg in state:
        if isinstance(msg, AIMessage):
            for tool_call in msg.tool_calls:
                name = tool_call.get("name")
                args = tool_call.get("args")
                dat = {}
                dat[name] = args
                workflow_dict["tool_calls"].append(dat)

    last_message = state[-1]
    content = getattr(last_message, "content", None)

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if "answer" in parsed:
                workflow_dict["result"] = parsed["answer"]
            else:
                workflow_dict["result"] = parsed
        except json.JSONDecodeError:
            workflow_dict["result"] = content
    else:
        workflow_dict["result"] = content

    return workflow_dict
