import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.tools.ase_tools import (
    run_ase,
    save_atomsdata_to_file,
    file_to_atomsdata,
)
from comp_chem_agent.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
)
from comp_chem_agent.prompt.multi_agent_prompt import (
    planner_prompt,
    executor_prompt,
    combiner_prompt,
    formatter_multi_prompt,
)
from comp_chem_agent.models.multi_agent_response import (
    PlannerResponse,
    ResponseFormatter,
)
from comp_chem_agent.utils.logging_config import setup_logger
from comp_chem_agent.state.multi_agent_state import ManagerWorkerState

logger = setup_logger(__name__)


class BasicToolNode:
    """A node that executes tools requested in the last AIMessage.

    This class processes tool calls from AI messages and executes the corresponding
    tools, handling their results and any potential errors. It maintains separate
    message channels for different workers.

    Parameters
    ----------
    tools : list
        List of tool objects that can be called by the node

    Attributes
    ----------
    tools_by_name : dict
        Dictionary mapping tool names to their corresponding tool objects
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: ManagerWorkerState) -> ManagerWorkerState:
        """Execute tools requested in the last message for the current worker.

        Parameters
        ----------
        inputs : ManagerWorkerState
            The current state containing worker channels and messages

        Returns
        -------
        ManagerWorkerState
            Updated state containing tool execution results

        Raises
        ------
        ValueError
            If no messages are found for the current worker
        """
        worker_id = inputs["current_worker"]

        # Access that worker's messages
        messages = inputs["worker_channel"].get(worker_id, [])
        if not messages:
            raise ValueError(f"No messages found for worker {worker_id}")

        message = messages[-1]  # Last AI message that called tools

        outputs = []

        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name")
                if not tool_name or tool_name not in self.tools_by_name:
                    raise ValueError(f"Invalid tool name: {tool_name}")

                tool_result = self.tools_by_name[tool_name].invoke(tool_call.get("args", {}))

                # Handle tool output: make it a dict or string
                result_content = (
                    tool_result.dict()
                    if hasattr(tool_result, "dict")
                    else (tool_result if isinstance(tool_result, dict) else str(tool_result))
                )

                outputs.append(
                    ToolMessage(
                        content=json.dumps(result_content),
                        name=tool_name,
                        tool_call_id=tool_call.get("id", ""),
                    )
                )

            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}),
                        name=tool_name if tool_name else "unknown_tool",
                        tool_call_id=tool_call.get("id", ""),
                    )
                )

        # Now, append the tool outputs directly into the worker's channel
        inputs["worker_channel"][worker_id].extend(outputs)

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


def PlannerAgent(state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str):
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
    structured_llm = llm.with_structured_output(PlannerResponse)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"messages": [response]}


def WorkerAgent(state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str, tools=None):
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
            file_to_atomsdata,
            smiles_to_atomsdata,
            run_ase,
            molecule_name_to_smiles,
            save_atomsdata_to_file,
        ]

    worker_id = state["current_worker"]
    history = state["worker_channel"].get(worker_id, [])

    messages = [{"role": "system", "content": system_prompt}] + history
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)

    # Append new LLM response directly back into the worker's channel
    state["worker_channel"][worker_id].append(response)

    # (optional) if no tool call, save it as worker_result
    if not getattr(response, "tool_calls", None):
        state["worker_result"] = [response]

    return state


def CombinerAgent(state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str):
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
    state: ManagerWorkerState, llm: ChatOpenAI, formatter_prompt: str = formatter_multi_prompt
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
    worker_id = task_list["worker_tasks"][task_idx].get("worker_id", f"worker_{task_idx}")

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


def contruct_multi_agent_graph(
    llm: ChatOpenAI,
    planner_prompt: str = planner_prompt,
    executor_prompt: str = executor_prompt,
    combiner_prompt: str = combiner_prompt,
    formatter_prompt: str = formatter_multi_prompt,
    structured_output: bool = False,
    tools: list = None,
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
    combiner_prompt : str, optional
        The prompt to guide result aggregation,
        by default combiner_prompt
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
        graph_builder.add_node(
            "PlannerAgent",
            lambda state: PlannerAgent(state, llm, system_prompt=planner_prompt),
        )
        graph_builder.add_node("extract_tasks", extract_tasks)
        graph_builder.add_node("loop_control", loop_control)

        graph_builder.add_node(
            "WorkerAgent",
            lambda state: WorkerAgent(state, llm, system_prompt=executor_prompt),
        )
        if tools is None:
            tools = [
                file_to_atomsdata,
                smiles_to_atomsdata,
                run_ase,
                molecule_name_to_smiles,
                save_atomsdata_to_file,
            ]

        graph_builder.add_node(
            "tools",
            BasicToolNode(tools=tools),
        )
        graph_builder.add_node("increment", increment_index)
        graph_builder.add_node(
            "CombinerAgent",
            lambda state: CombinerAgent(state, llm, system_prompt=combiner_prompt),
        )
        graph_builder.add_conditional_edges(
            "loop_control",
            worker_iterator,
            {"worker": "WorkerAgent", "aggregate": "CombinerAgent"},
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
            graph_builder.add_edge("CombinerAgent", END)
        else:
            graph_builder.add_node(
                "ResponseAgent",
                lambda state: ResponseAgent(state, llm, formatter_prompt=formatter_prompt),
            )
            graph_builder.add_edge("CombinerAgent", "ResponseAgent")
            graph_builder.add_edge("ResponseAgent", END)

        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Graph construction completed")
        return graph

    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
