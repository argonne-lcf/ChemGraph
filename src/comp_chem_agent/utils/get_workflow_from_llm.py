import json
from langchain.schema.messages import AIMessage


def get_workflow_from_log(file_path: str) -> dict:
    """Convert a run_logs file to a workflow dictionary for evaluations

    Args:
        file_path (str): File path to run logs

    Returns:
        dict: a workflow dictionary
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    # Extract tool names and arguments
    workflow_dict = {'tool_calls': []}
    for state in data.get('state', []):
        tool_calls = state.get('tool_calls', [])
        for call in tool_calls:
            name = call.get('name')
            args = call.get('args')
            dat = {}
            dat[name] = args
            workflow_dict['tool_calls'].append(args)
    last_message = data.get('state', [])[-1]
    try:
        if "answer" in last_message['content']:
            result_data = json.loads(last_message['content'])
            workflow_dict['result'] = result_data.get('answer')
    except Exception as e:
        result_data = last_message['content']
        workflow_dict['result'] = result_data

    return workflow_dict


def get_workflow_from_state(state) -> dict:
    workflow_dict = {'tool_calls': []}
    for msg in state:
        if isinstance(msg, AIMessage):
            for tool_call in msg.tool_calls:
                name = tool_call.get('name')
                args = tool_call.get('args')
                dat = {}
                dat[name] = args
                workflow_dict['tool_calls'].append(dat)

    last_message = state[-1]
    content = getattr(last_message, 'content', None)

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if 'answer' in parsed:
                workflow_dict['result'] = parsed['answer']
            else:
                workflow_dict['result'] = parsed
        except json.JSONDecodeError:
            workflow_dict['result'] = content
    else:
        workflow_dict['result'] = content

    return workflow_dict
