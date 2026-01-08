import json
from typing import Any
from pydantic import BaseModel

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from chemgraph.tools.ase_tools import run_ase, extract_output_json
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_coordinate_file,
)
from chemgraph.prompt.multi_agent_prompt import (
    planner_prompt,
    executor_prompt,
    aggregator_prompt,
    formatter_multi_prompt,
    planner_prompt_json,
)
from chemgraph.schemas.multi_agent_response import (
    PlannerResponse,
    ResponseFormatter,
)
from chemgraph.utils.logging_config import setup_logger
from chemgraph.state.multi_agent_state import ManagerWorkerState

logger = setup_logger(__name__)


### Help Functions
def _to_jsonable(obj: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    else:
        return obj


def sanitize_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure tool_call['args'] contains only JSON-serializable data."""
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            new_tool_calls = []
            for tc in m.tool_calls:
                tc = dict(tc)
                tc["args"] = _to_jsonable(tc.get("args"))
                new_tool_calls.append(tc)
            m.tool_calls = new_tool_calls
    return messages


def route_tools(state: ManagerWorkerState):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing worker channels and messages

    Returns
    -------
    str
        Either 'tools' or 'done' based on the presence of tool calls

    Raises
    ------
    ValueError
        If no messages are found for the current worker
    """
    worker_id = state["current_worker"]
    messages = state.get("worker_messages", [])

    if not messages:
        raise ValueError(
            f"No messages found for worker {worker_id} in worker_messages."
        )

    ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    return "done"


def PlannerAgent(
    state: ManagerWorkerState,
    llm: ChatOpenAI,
    system_prompt: str,
    support_structured_output: bool,
):
    """An LLM agent that decomposes tasks into subtasks for workers.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing the task to be decomposed
    llm : ChatOpenAI
        The language model to use for task decomposition
    system_prompt : str
        The system prompt to guide the task decomposition

    Returns
    -------
    dict
        Updated state containing the decomposed tasks
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    if support_structured_output is True:
        structured_llm = llm.with_structured_output(PlannerResponse)
        response = structured_llm.invoke(messages).model_dump_json()
        return {"messages": [response]}
    else:
        raw_response = llm.invoke(messages).content
        try:
            # Always parse from string
            parsed = json.loads(raw_response)

            # If model mistakenly returned a bare list, wrap it
            if isinstance(parsed, list):
                parsed = {"worker_tasks": parsed}

            # Validate structure
            if not isinstance(parsed, dict) or "worker_tasks" not in parsed:
                raise ValueError(
                    "Output must be a JSON object with a 'worker_tasks' key"
                )

            response_json = json.dumps(parsed)
            return {"messages": [response_json]}

        except Exception as e:
            retry_message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{state.get('messages', '')}"},
                {
                    "role": "assistant",
                    "content": (
                        f"Error: {str(e)}. Please output a valid JSON object with a 'worker_tasks' key, "
                        "where 'worker_tasks' is a list of tasks in the format:\n"
                        '{"worker_tasks": [\n'
                        '  {"task_index": 1, "prompt": "Calculate ..."},\n'
                        '  {"task_index": 2, "prompt": "Calculate ..."}\n'
                        ']}'
                    ),
                },
            ]
            retry_response = llm.invoke(retry_message).content
            parsed_retry = json.loads(retry_response)

            # Normalize again in case it’s a list
            if isinstance(parsed_retry, list):
                parsed_retry = {"worker_tasks": parsed_retry}

            response_json = json.dumps(parsed_retry)
            return {"messages": [response_json]}


def WorkerAgent(
    state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str, tools=None
):
    """An LLM agent that executes assigned tasks using available tools.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing worker channels and task information
    llm : ChatOpenAI
        The language model to use for task execution
    system_prompt : str
        The system prompt to guide the worker's behavior
    tools : list, optional
        List of tools available to the worker, by default None

    Returns
    -------
    ManagerWorkerState
        Updated state containing the worker's response and results
    """
    if tools is None:
        tools = [
            run_ase,
            molecule_name_to_smiles,
            smiles_to_coordinate_file,
            extract_output_json,
        ]

    worker_id = state["current_worker"]
    history = state.get("worker_messages", [])

    history = sanitize_tool_calls(history)

    messages = [{"role": "system", "content": system_prompt}] + history
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)

    # Append new LLM response directly back into the worker's channel
    state["worker_messages"].append(response)
    state["worker_channel"][worker_id] = state["worker_messages"]

    # If no tool call, save it as worker_result
    if not getattr(response, "tool_calls", None):
        state["worker_result"] = [response]

    return state


def AggregatorAgent(state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str):
    """An LLM agent that aggregates results from all workers.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing worker results
    llm : ChatOpenAI
        The language model to use for result aggregation
    system_prompt : str
        The system prompt to guide the aggregation process

    Returns
    -------
    dict
        Updated state containing the aggregated results
    """
    if "worker_result" in state:
        outputs = [m.content for m in state["worker_result"]]
        worker_summary_msg = {
            "role": "assistant",
            "content": "Worker Outputs:\n" + "\n".join(outputs),
        }
        state["messages"].append(worker_summary_msg)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}


def ResponseAgent(
    state: ManagerWorkerState,
    llm: ChatOpenAI,
    formatter_prompt: str = formatter_multi_prompt,
):
    """An LLM agent responsible for formatting the final response.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing the aggregated results
    llm : ChatOpenAI
        The language model to use for response formatting
    formatter_prompt : str, optional
        The prompt to guide the formatting process,
        by default formatter_prompt

    Returns
    -------
    dict
        Updated state containing the formatted response
    """
    messages = [
        {"role": "system", "content": formatter_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_structured_output = llm.with_structured_output(ResponseFormatter)
    response = llm_structured_output.invoke(messages).model_dump_json()
    return {"messages": [response]}


def extract_tasks(state: ManagerWorkerState):
    """Extract task list from the task decomposer's response.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing the task decomposer's response

    Returns
    -------
    ManagerWorkerState
        Updated state with extracted task list and initialized task index
    """
    state["task_list"] = state["messages"][-1].content
    state["current_task_index"] = 0
    return state


def loop_control(state: ManagerWorkerState):
    """Prepare the next task for the current worker.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing task list and worker information

    Returns
    -------
    ManagerWorkerState
        Updated state with prepared task for the current worker
    """
    task_idx = state["current_task_index"]
    task_list = json.loads(state["task_list"])

    # If finished all tasks, do nothing. worker_iterator will handle it
    if task_idx >= len(task_list["worker_tasks"]):
        return state

    task_prompt = task_list["worker_tasks"][task_idx]["prompt"]
    worker_id = task_list["worker_tasks"][task_idx].get(
        "worker_id", f"worker_{task_idx}"
    )

    state["current_worker"] = worker_id

    if "worker_channel" not in state:
        state["worker_channel"] = {}

    if worker_id not in state["worker_channel"]:
        state["worker_channel"][worker_id] = []

    state["worker_channel"][worker_id].append({"role": "user", "content": task_prompt})
    state["worker_messages"] = state["worker_channel"][worker_id]

    print(f"[Worker {worker_id}] Now processing task: '{task_prompt}'")
    return state


def worker_iterator(state: ManagerWorkerState):
    """Determine the next step in the workflow based on task completion.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing task list and progress

    Returns
    -------
    str
        Either 'aggregate' if all tasks are done, or 'worker' to continue with tasks
    """
    task_idx = state["current_task_index"]
    task_list = json.loads(state["task_list"])

    if task_idx >= len(task_list["worker_tasks"]):
        return "aggregate"
    else:
        return "worker"


def increment_index(state: ManagerWorkerState):
    """Increment the current task index.

    Parameters
    ----------
    state : ManagerWorkerState
        The current state containing task progress

    Returns
    -------
    ManagerWorkerState
        Updated state with incremented task index
    """
    state["current_task_index"] += 1
    return state


def contruct_multi_agent_graph(
    llm: ChatOpenAI,
    planner_prompt: str = planner_prompt,
    executor_prompt: str = executor_prompt,
    aggregator_prompt: str = aggregator_prompt,
    formatter_prompt: str = formatter_multi_prompt,
    structured_output: bool = False,
    tools: list = None,
    support_structured_output: bool = True,
):
    """Construct a graph for manager-worker workflow.

    This function creates a state graph that implements a manager-worker pattern
    for computational chemistry tasks, where tasks are decomposed and executed
    by specialized workers.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use in the workflow
    planner_prompt : str, optional
        The prompt to guide task decomposition,
        by default planner_prompt
    executor_prompt : str, optional
        The prompt to guide worker behavior,
        by default executor_prompt
    aggregator_prompt : str, optional
        The prompt to guide result aggregation,
        by default aggregator_prompt
    structured_output : bool, optional
        Whether to use structured output format,
        by default False
    tools: list, optional
        The tools provided for the agent,
        by default None

    Returns
    -------
    StateGraph
        A compiled state graph implementing the manager-worker workflow

    Raises
    ------
    Exception
        If there is an error during graph construction
    """
    try:
        logger.info("Constructing multi-agent graph")
        checkpointer = MemorySaver()

        graph_builder = StateGraph(ManagerWorkerState)
        if support_structured_output is True:
            graph_builder.add_node(
                "PlannerAgent",
                lambda state: PlannerAgent(
                    state,
                    llm,
                    system_prompt=planner_prompt,
                    support_structured_output=support_structured_output,
                ),
            )
        else:
            graph_builder.add_node(
                "PlannerAgent",
                lambda state: PlannerAgent(
                    state,
                    llm,
                    system_prompt=planner_prompt_json,
                    support_structured_output=support_structured_output,
                ),
            )

        graph_builder.add_node("extract_tasks", extract_tasks)
        graph_builder.add_node("loop_control", loop_control)

        graph_builder.add_node(
            "WorkerAgent",
            lambda state: WorkerAgent(state, llm, system_prompt=executor_prompt),
        )
        if tools is None:
            tools = [
                run_ase,
                molecule_name_to_smiles,
                smiles_to_coordinate_file,
                extract_output_json,
            ]
        tools_node = ToolNode(tools=tools, messages_key="worker_messages")
        graph_builder.add_node("tools", tools_node)
        graph_builder.add_node("increment", increment_index)
        graph_builder.add_node(
            "AggregatorAgent",
            lambda state: AggregatorAgent(state, llm, system_prompt=aggregator_prompt),
        )
        graph_builder.add_conditional_edges(
            "loop_control",
            worker_iterator,
            {"worker": "WorkerAgent", "aggregate": "AggregatorAgent"},
        )
        graph_builder.set_entry_point("PlannerAgent")
        graph_builder.add_edge("PlannerAgent", "extract_tasks")
        graph_builder.add_edge("extract_tasks", "loop_control")
        graph_builder.add_edge("tools", "WorkerAgent")
        graph_builder.add_conditional_edges(
            "WorkerAgent",
            route_tools,
            {"tools": "tools", "done": "increment"},
        )

        graph_builder.add_edge("increment", "loop_control")

        if not structured_output:
            graph_builder.add_edge("AggregatorAgent", END)
        else:
            graph_builder.add_node(
                "ResponseAgent",
                lambda state: ResponseAgent(
                    state, llm, formatter_prompt=formatter_prompt
                ),
            )
            graph_builder.add_edge("AggregatorAgent", "ResponseAgent")
            graph_builder.add_edge("ResponseAgent", END)

        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Graph construction completed")
        return graph

    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
