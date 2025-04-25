from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage
import json
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.tools.ASE_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
    run_ase,
    save_atomsdata_to_file,
    file_to_atomsdata,
)
from comp_chem_agent.tools.generic import calculator
from comp_chem_agent.prompt.manager_worker_prompt import (
    task_decomposer_prompt,
    worker_prompt,
    result_aggregator_prompt,
)
from comp_chem_agent.models.manager_worker_response import TaskDecomposerResponse
from comp_chem_agent.utils.logging_config import setup_logger
from comp_chem_agent.state.manager_worker_state import ManagerWorkerState

logger = setup_logger(__name__)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: ManagerWorkerState) -> ManagerWorkerState:
        if messages := inputs.get("worker_state", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name")
                if not tool_name or tool_name not in self.tools_by_name:
                    raise ValueError(f"Invalid tool name: {tool_name}")

                tool_result = self.tools_by_name[tool_name].invoke(tool_call.get("args", {}))

                # Handle different types of tool results
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
        return {"worker_state": outputs}


def route_tools(state: ManagerWorkerState):
    """
    Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("worker_state", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in worker_state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


def TaskDecomposerAgent(state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str):
    """Task Decomposer Agent"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    structured_llm = llm.with_structured_output(TaskDecomposerResponse)
    response = structured_llm.invoke(messages).model_dump_json()

    print("*****TASK DECOMPOSER*****")
    print(f'messages: {messages}')
    print(f'response: {response}')
    return {"messages": [response]}


def WorkerAgent(state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str, tools=None):
    """Worker Agent"""
    if tools is None:
        tools = [
            file_to_atomsdata,
            smiles_to_atomsdata,
            run_ase,
            molecule_name_to_smiles,
            save_atomsdata_to_file,
            calculator,
        ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['worker_state']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)
    result = {"worker_state": [response]}

    # If no tool call was made, return the output directly too
    if not getattr(response, "tool_calls", None):
        result["worker_output"] = [response]
    print("*****WORKER*****")
    print(f'messages: {messages}')
    print(f'response: {response}')

    return result


def ResultAggregatorAgent(state: ManagerWorkerState, llm: ChatOpenAI, system_prompt: str):
    """Result Aggregator Agent"""
    messages = [
        {"role": "system", "content": system_prompt.format(state['worker_output'])},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    response = llm.invoke(messages)
    print("*****ResultAggregatorAgent*****")
    print(f'messages: {messages}')
    print(f'response: {response}')

    return {"messages": [response]}


# -----------------------------
# Graph Logic
# -----------------------------
def extract_tasks(state: ManagerWorkerState):
    print("******EXTRACT TASK******")
    print(state)
    state["task_list"] = state["messages"][-1].content
    state["current_task_index"] = 0
    return state


def loop_control(state: ManagerWorkerState):
    print("*********************************")
    print("LOOP CONTROL")
    print("*********************************")
    task_idx = state["current_task_index"]
    task_list = json.loads(state["task_list"])

    if task_idx >= len(task_list["worker_tasks"]):
        return state

    print(f"OLD STATE: {state['worker_state']}")
    state.pop("worker_state", None)

    task_prompt = task_list["worker_tasks"][task_idx]["prompt"]
    state["worker_state"] = [{"role": "user", "content": task_prompt}]

    print(f"NEW STATE: {state['worker_state']}")
    return state


def worker_iterator(state: ManagerWorkerState):
    if state["current_task_index"] >= len(state["task_list"]):
        return "aggregate"
    return "worker"


def increment_index(state: ManagerWorkerState):
    state["current_task_index"] += 1
    return state


# -----------------------------
# Assemble LangGraph
# -----------------------------


def contruct_manager_worker_graph(
    llm: ChatOpenAI,
    task_decomposer_prompt: str = task_decomposer_prompt,
    worker_prompt: str = worker_prompt,
    result_aggregator_prompt: str = result_aggregator_prompt,
):
    try:
        logger.info("Constructing geometry optimization graph")
        checkpointer = MemorySaver()
        graph_builder = StateGraph(ManagerWorkerState)

        graph_builder.add_node(
            "decompose",
            lambda state: TaskDecomposerAgent(state, llm, system_prompt=task_decomposer_prompt),
        )
        graph_builder.add_node("extract_tasks", extract_tasks)
        graph_builder.add_node("loop_control", loop_control)

        graph_builder.add_node(
            "worker", lambda state: WorkerAgent(state, llm, system_prompt=worker_prompt)
        )
        graph_builder.add_node(
            "tools",
            BasicToolNode([
                molecule_name_to_smiles,
                smiles_to_atomsdata,
                run_ase,
                save_atomsdata_to_file,
                file_to_atomsdata,
                calculator,
            ]),
        )
        graph_builder.add_node("increment", increment_index)
        graph_builder.add_node(
            "aggregate",
            lambda state: ResultAggregatorAgent(state, llm, system_prompt=result_aggregator_prompt),
        )
        graph_builder.add_conditional_edges(
            "loop_control",
            worker_iterator,
            {"worker": "worker", "aggregate": "aggregate"},
        )
        graph_builder.set_entry_point("decompose")
        graph_builder.add_edge("decompose", "extract_tasks")
        graph_builder.add_edge("extract_tasks", "loop_control")
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "worker",
            route_tools,
            {"tools": "tools", "done": "increment"},
        )

        graph_builder.add_edge("increment", "loop_control")
        graph_builder.add_edge("aggregate", END)

        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Graph construction completed")
        return graph

    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
