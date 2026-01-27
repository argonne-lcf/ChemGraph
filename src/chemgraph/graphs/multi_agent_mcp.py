import json
from typing import List, Optional, Any, Dict
import inspect

from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage
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


class AsyncBasicToolNode:
    """A node that executes tools requested in the last AIMessage.

    This class processes tool calls from AI messages and executes the corresponding
    tools, handling their results and any potential errors. It maintains separate
    message channels for different workers.
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def _json_sanitize(self, x):
        """Recursively convert Pydantic BaseModel and other
        nested objects to JSON-serializable types."""
        if isinstance(x, BaseModel):
            # pydantic v2 preferred
            try:
                return self._json_sanitize(x.model_dump(exclude_none=True))
            except Exception:
                # pydantic v1 fallback
                return self._json_sanitize(x.dict(exclude_none=True))
        if isinstance(x, dict):
            return {k: self._json_sanitize(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self._json_sanitize(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        # Last resort: stringify unknown objects
        return str(x)

    async def _run_tool(self, tool: Any, args_obj: Dict[str, Any]) -> Any:
        """Call tool asynchronously: prefer .ainvoke; else await the callable if it's a coroutine function."""
        if hasattr(tool, "ainvoke") and callable(tool.ainvoke):
            return await tool.ainvoke(args_obj)
        # allow async function tools (e.g., @tool on async def)
        if inspect.iscoroutinefunction(tool):
            return await tool(**args_obj)
        # Optional: support wrappers exposing .invoke that return awaitables
        if hasattr(tool, "invoke") and callable(tool.invoke):
            maybe = tool.invoke(args_obj)
            if inspect.isawaitable(maybe):
                return await maybe
        raise NotImplementedError(
            f"Tool '{getattr(tool, 'name', tool)}' is not async (.ainvoke or async def required)."
        )

    async def __call__(self, inputs):
        worker_id = inputs["current_worker"]
        channel = inputs.setdefault("worker_channel", {})
        history = channel.get(worker_id, [])
        if not history:
            raise ValueError(f"No messages found for worker {worker_id}")

        # Last assistant msg should contain tool_calls
        message = history[-1]
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return inputs

        outputs: List[ToolMessage] = []

        for tc in tool_calls:
            tool_name: Optional[str] = None
            try:
                if "function" in tc:
                    fn = tc["function"]
                    tool_name = fn.get("name")
                    raw_args = fn.get("arguments", {})
                else:
                    tool_name = tc.get("name")
                    raw_args = tc.get("args", {})

                if not tool_name or tool_name not in self.tools_by_name:
                    raise ValueError(f"Invalid tool name: {tool_name}")

                # Parse/normalize args
                if isinstance(raw_args, str):
                    try:
                        args_obj = json.loads(raw_args) if raw_args.strip() else {}
                    except Exception:
                        args_obj = {}
                else:
                    args_obj = raw_args or {}

                args_obj = self._json_sanitize(args_obj)

                tool = self.tools_by_name[tool_name]
                result = await self._run_tool(tool, args_obj)

                # Normalize result to JSON-ish content
                if hasattr(result, "model_dump"):
                    payload = result.model_dump()
                elif hasattr(result, "dict"):
                    payload = result.dict()
                elif isinstance(result, dict):
                    payload = result
                else:
                    payload = {"result": str(result)}

                outputs.append(
                    ToolMessage(
                        content=json.dumps(payload),
                        name=tool_name,
                        tool_call_id=tc.get("id", tc.get("function", {}).get("id", "")),
                    )
                )

            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}),
                        name=tool_name or "unknown_tool",
                        tool_call_id=tc.get("id", tc.get("function", {}).get("id", "")),
                    )
                )

        channel.setdefault(worker_id, []).extend(outputs)
        return inputs


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
    if messages := state["worker_channel"].get(worker_id, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found for worker {worker_id} in worker_channel.")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


def PlannerAgent(
    state: ManagerWorkerState,
    llm: ChatOpenAI,
    system_prompt: str,
    support_structured_output: bool,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    if support_structured_output:
        structured_llm = llm.with_structured_output(PlannerResponse)
        response = structured_llm.invoke(messages)
        return {"messages": [response.model_dump_json()]}
    else:
        raw_response = (llm.invoke(messages)).content
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, list):
                parsed = {"worker_tasks": parsed}
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
            retry_response = (llm.invoke(retry_message)).content
            parsed_retry = json.loads(retry_response)
            if isinstance(parsed_retry, list):
                parsed_retry = {"worker_tasks": parsed_retry}
            response_json = json.dumps(parsed_retry)
            return {"messages": [response_json]}


def WorkerAgent(
    state: ManagerWorkerState,
    llm: ChatOpenAI,
    system_prompt: str,
    tools: list = None,
):
    if tools is None:
        tools = [
            run_ase,
            molecule_name_to_smiles,
            smiles_to_coordinate_file,
            extract_output_json,
        ]

    worker_id = state["current_worker"]
    history = state["worker_channel"].get(worker_id, [])

    messages = [{"role": "system", "content": system_prompt}] + history
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)
    print(messages)
    print(tools)
    print(list(tools))
    print(response)
    state["worker_channel"][worker_id].append(response)

    if not getattr(response, "tool_calls", None):
        state["worker_result"] = [response]

    return state


def AggregatorAgent(
    state: ManagerWorkerState,
    llm: ChatOpenAI,
    system_prompt: str,
):
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
    messages = [
        {"role": "system", "content": formatter_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_structured_output = llm.with_structured_output(ResponseFormatter)
    response = llm_structured_output.invoke(messages)
    return {"messages": [response.model_dump_json()]}


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


def construct_multi_agent_mcp_graph(
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

        graph_builder.add_node(
            "tools",
            AsyncBasicToolNode(tools=tools),
        )
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
