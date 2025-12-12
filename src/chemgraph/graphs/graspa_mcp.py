from functools import partial

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send

from chemgraph.utils.logging_config import setup_logger
from chemgraph.state.graspa_state import (
    ExecutorState,
    PlannerState,
    PlannerResponse,
)
from chemgraph.prompt.graspa_prompt import (
    planner_prompt,
    executor_prompt,
    aggregator_prompt,
)

logger = setup_logger(__name__)


def planner_agent(
    state: PlannerState,
    llm: ChatOpenAI,
    system_prompt: str,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    structured_llm = llm.with_structured_output(PlannerResponse)
    response_obj = structured_llm.invoke(messages)

    return {"messages": [response_obj.model_dump_json()], "tasks": response_obj.tasks}


async def executor_model_node(
    state: ExecutorState,
    llm: ChatOpenAI,
    system_prompt: str,
    tools: list,
):
    """
    The reasoning engine for a single executor.
    It sees its own 'task_prompt' and its own 'messages' history.
    """
    messages = state["messages"]

    # If this is the first step, prepend the System Prompt and the Task
    if not messages:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Your Task: {state['task_prompt']}"},
        ]

    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


def route_executor(state: ExecutorState):
    """Standard ReAct routing: Tool vs End."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "done"


def format_executor_output(state: ExecutorState) -> PlannerState:
    """
    Bridge function:
    Converts the Local ExecutorState into an update for the Global PlannerState.
    """
    executor_id = state["executor_id"]
    final_message = state["messages"][-1].content
    full_history = state["messages"]

    return {
        "executor_results": [f"[{executor_id}] Result: {final_message}"],
        "executor_logs": {executor_id: full_history},
    }


def distribute_tasks(state: PlannerState):
    """
    Iterates over the list of tasks and creates a `Send` object for each.
    Each `Send` targets the 'executor_graph' node with a specific input state.
    """
    tasks = state.get("tasks", [])
    sends = []

    for i, task in enumerate(tasks):
        executor_id = getattr(task, "executor_id", f"executor_{i}")
        prompt = getattr(task, "prompt", "Analyze this.")

        payload = {
            "executor_id": executor_id,
            "task_prompt": prompt,
            "messages": [],
        }

        sends.append(Send("executor_subgraph", payload))

    return sends


def aggreagator_agent(
    state: PlannerState,
    llm: ChatOpenAI,
    system_prompt: str,
):
    """Synthesizes all collected results from state['executor_results']."""
    results = state.get("executor_results", [])

    formatted_results = "\n\n".join(
        [f"Report {i + 1} ---\n{res}" for i, res in enumerate(results)]
    )
    summary_query = (
        "All executor tasks are complete. Here are their collected reports:\n\n"
        f"{formatted_results}\n\n"
        "Please synthesize these results to answer the original user request."
    )
    state["messages"].append(summary_query)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]

    response = llm.invoke(messages)
    return {"messages": [response]}


def construct_executor_subgraph(llm: ChatOpenAI, tools: list, system_prompt: str):
    """Builds the reusable executor subgraph (Agent -> Tools -> Agent)."""
    workflow = StateGraph(ExecutorState)
    workflow.add_node(
        "executor_agent",
        partial(executor_model_node, llm=llm, system_prompt=system_prompt, tools=tools),
    )
    workflow.add_node("tools", ToolNode(tools))

    # Format the output back to the Main State schema
    workflow.add_node("finalize", format_executor_output)

    # Edges
    workflow.set_entry_point("executor_agent")
    workflow.add_conditional_edges(
        "executor_agent",
        route_executor,
        {"tools": "tools", "done": "finalize"},
    )
    workflow.add_edge("tools", "executor_agent")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def contruct_graspa_mcp_graph(
    llm: ChatOpenAI,
    planner_prompt: str = planner_prompt,
    executor_prompt: str = executor_prompt,
    aggregator_prompt: str = aggregator_prompt,
    tools: list = None,
):
    """
    Constructs the Main Graph using the Map-Reduce (Send) pattern.
    """
    checkpointer = MemorySaver()

    # Create the Executor subgraph
    executor_subgraph = construct_executor_subgraph(llm, tools, executor_prompt)

    # Create the Main Manager Graph
    graph_builder = StateGraph(PlannerState)

    # Planner agent
    graph_builder.add_node(
        "Planner",
        lambda state: planner_agent(
            state,
            llm,
            planner_prompt,
        ),
    )

    # Executor agent
    graph_builder.add_node("executor_subgraph", executor_subgraph)

    # Aggregator agent
    graph_builder.add_node(
        "Aggregator",
        lambda state: aggreagator_agent(
            state,
            llm,
            aggregator_prompt,
        ),
    )

    # -- Edges --
    graph_builder.set_entry_point("Planner")

    # Conditional Edge: Planner -> [Send(executor)...]
    graph_builder.add_conditional_edges(
        "Planner",
        distribute_tasks,
        ["executor_subgraph"],
    )
    graph_builder.add_edge("executor_subgraph", "Aggregator")
    graph_builder.add_edge("Aggregator", END)

    return graph_builder.compile(checkpointer=checkpointer)
