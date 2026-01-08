from typing import Union
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
    batch_orchestrator_prompt,
    analyst_prompt,
)

logger = setup_logger(__name__)


def planner_agent(
    state: PlannerState,
    llm: ChatOpenAI,
    system_prompt: str,
):
    executor_outputs = state.get("executor_results", [])
    content_block = f"Current Conversation History: {state['messages']}"
    if executor_outputs:
        results_text = "\n".join(executor_outputs)
        content_block += (
            f"\n\n### UPDATED: Results from Executor Tasks ###\n{results_text}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": content_block},
    ]

    structured_llm = llm.with_structured_output(PlannerResponse)
    response_obj = structured_llm.invoke(messages)
    print(f"PLANNER: {response_obj.model_dump_json()}")
    return {
        "messages": [response_obj.thought_process],
        "next_step": response_obj.next_step,
        "tasks": response_obj.tasks if response_obj.tasks else [],
    }


def unified_planner_router(state: PlannerState) -> Union[str, list[Send]]:
    """
    Routes based on the Planner's structured 'next_step'.
    """
    next_step = state.get("next_step")

    # --- PATH A: PARALLEL EXECUTION ---
    if next_step == "executor_subgraph":
        tasks = state.get("tasks", [])
        return [
            Send(
                "executor_subgraph",
                {
                    "executor_id": f"worker_{getattr(t, 'task_index', i + 1)}",
                    "messages": [getattr(t, 'prompt')],
                },
            )
            for i, t in enumerate(tasks)
        ]

    # --- PATH B: STANDARD ROUTING ---
    elif next_step == "batch_orchestrator":
        return "batch_orchestrator"

    elif next_step == "insight_analyst":
        return "insight_analyst"

    # --- PATH D: TERMINATION ---
    return END


def batch_orchestrator_agent(
    state: PlannerState, llm: ChatOpenAI, tools: list, system_prompt: str
):
    """Decides to call the split tool."""
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def route_batch_orchestrator(state: PlannerState):
    """
    If the agent called a tool, go to ToolNode.
    Otherwise (or after tool execution), go back to Planner to decide next steps.
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "batch_tools"

    return "Planner"


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
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)

    # print(f"EXECUTOR: {response}")
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


def insight_analyst_node(
    state: PlannerState,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
):
    """Analyzes the gathered results."""
    results_text = "\n".join(state["executor_results"])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": f"Previous Messages: {state['messages']}\n\nExecution Results:\n{results_text}",
        },
    ]

    # Simple single-step analyst (can be expanded to loop like Executor)
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}


def route_analyst(state: PlannerState):
    """
    Determines if the Analyst is calling a tool or giving the final answer.
    """
    last_msg = state["messages"][-1]

    # If the Analyst wants to use Python/Pandas -> Go to ToolNode
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "analyst_tools"

    # If the Analyst is done -> Go back to Planner (who will then trigger FINISH)
    return "Planner"


def contruct_graspa_mcp_graph(
    llm: ChatOpenAI,
    planner_prompt: str = planner_prompt,
    executor_prompt: str = executor_prompt,
    batch_orchestrator_prompt: str = batch_orchestrator_prompt,
    analyst_prompt: str = analyst_prompt,
    executor_tools: list = None,
    analysis_tools: list = None,
):
    """
    Constructs the Main Graph using the Map-Reduce (Send) pattern.
    """
    checkpointer = MemorySaver()

    # Create the Executor subgraph
    executor_subgraph = construct_executor_subgraph(
        llm,
        executor_tools,
        executor_prompt,
    )

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

    # batch_orchestrator agent
    graph_builder.add_node(
        "batch_orchestrator",
        partial(
            batch_orchestrator_agent,
            llm=llm,
            tools=analysis_tools,
            system_prompt=batch_orchestrator_prompt,
        ),
    )
    # Executor agent
    graph_builder.add_node("executor_subgraph", executor_subgraph)

    # Analyst agent
    graph_builder.add_node(
        "insight_analyst",
        partial(
            insight_analyst_node,
            llm=llm,
            tools=analysis_tools,
            system_prompt=analyst_prompt,
        ),
    )
    # Tool nodes
    graph_builder.add_node("batch_tools", ToolNode(analysis_tools))
    graph_builder.add_node("analyst_tools", ToolNode(analysis_tools))

    # -- Edges --
    graph_builder.set_entry_point("Planner")

    # Conditional Edge: Planner -> [Send(executor)...]
    graph_builder.add_conditional_edges(
        "Planner",
        unified_planner_router,
        [
            "batch_orchestrator",
            "insight_analyst",
            "executor_subgraph",
            END,
        ],
    )
    graph_builder.add_conditional_edges(
        "batch_orchestrator",
        route_batch_orchestrator,
        {"batch_tools": "batch_tools", "Planner": "Planner"},
    )
    graph_builder.add_edge("batch_tools", "batch_orchestrator")
    graph_builder.add_edge("executor_subgraph", "Planner")
    graph_builder.add_conditional_edges(
        "insight_analyst",
        route_analyst,
        {
            "analyst_tools": "analyst_tools",
            "Planner": END,
        },
    )
    graph_builder.add_edge("analyst_tools", "insight_analyst")

    return graph_builder.compile(checkpointer=checkpointer)
