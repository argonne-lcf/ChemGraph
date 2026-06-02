from typing import Union
from functools import partial

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from chemgraph.utils.logging_config import setup_logger
from chemgraph.state.graspa_state import (
    ExecutorState,
    PlannerState,
    PlannerResponse,
)
from chemgraph.prompt.graspa_prompt import (
    planner_prompt,
    executor_prompt,
    analyst_prompt,
)

logger = setup_logger(__name__)


def planner_agent(
    state: PlannerState,
    llm: ChatOpenAI,
    system_prompt: str,
):
    """Plan the next gRASPA MCP workflow step.

    Parameters
    ----------
    state : PlannerState
        Current planner state containing messages and executor results.
    llm : ChatOpenAI
        Chat model used for planning.
    system_prompt : str
        Planner system prompt.

    Returns
    -------
    dict
        Planner state update containing messages, next step, and tasks.
    """
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
    """Route based on the planner's structured ``next_step``.

    Parameters
    ----------
    state : PlannerState
        Current planner state.

    Returns
    -------
    str or list[Send]
        Next node name, ``END``, or fan-out executor sends.
    """
    next_step = state.get("next_step")

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

    elif next_step == "insight_analyst":
        return "insight_analyst"

    return END


async def executor_model_node(
    state: ExecutorState,
    llm: ChatOpenAI,
    system_prompt: str,
    tools: list,
):
    """Run the reasoning step for a single gRASPA executor.

    Parameters
    ----------
    state : ExecutorState
        Local executor state.
    llm : ChatOpenAI
        Chat model used by the executor.
    system_prompt : str
        Executor system prompt.
    tools : list
        Tools available to the executor.

    Returns
    -------
    dict
        Executor state update containing the model response.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    # Flatten MCP/LangChain content blocks to plain text before ChatOpenAI
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, list):
            text = "\n".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content
            )
            if isinstance(m, dict):
                m["content"] = text
            else:
                m.content = text

    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}

def route_executor(state: ExecutorState):
    """Route executor output to tools or completion.

    Parameters
    ----------
    state : ExecutorState
        Local executor state.

    Returns
    -------
    str
        ``"tools"`` when tool calls are present, otherwise ``"done"``.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "done"


def format_executor_output(state: ExecutorState) -> PlannerState:
    """Convert local executor state into a global planner update.

    Parameters
    ----------
    state : ExecutorState
        Local executor state at subgraph completion.

    Returns
    -------
    PlannerState
        Planner update containing executor results and logs.
    """
    executor_id = state["executor_id"]
    final_message = state["messages"][-1].content
    full_history = state["messages"]

    return {
        "executor_results": [f"[{executor_id}] Result: {final_message}"],
        "executor_logs": {executor_id: full_history},
    }


def construct_executor_subgraph(llm: ChatOpenAI, tools: list, system_prompt: str):
    """Build the reusable executor subgraph.

    Parameters
    ----------
    llm : ChatOpenAI
        Chat model used by executor agents.
    tools : list
        Tools available to executor agents.
    system_prompt : str
        Executor system prompt.

    Returns
    -------
    CompiledStateGraph
        Compiled executor subgraph.
    """
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
    """Analyze gathered executor results.

    Parameters
    ----------
    state : PlannerState
        Planner state containing executor results.
    llm : ChatOpenAI
        Chat model used by the analyst.
    tools : list
        Analysis tools available to the analyst.
    system_prompt : str
        Analyst system prompt.

    Returns
    -------
    dict
        Planner state update containing the analyst response.
    """
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
    """Route analyst output to tools or back to the planner.

    Parameters
    ----------
    state : PlannerState
        Planner state containing the analyst's latest message.

    Returns
    -------
    str
        ``"analyst_tools"`` when tool calls are present, otherwise
        ``"Planner"``.
    """
    last_msg = state["messages"][-1]

    # If the Analyst wants to use Python/Pandas -> Go to ToolNode
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "analyst_tools"

    # If the Analyst is done -> Go back to Planner (who will then trigger FINISH)
    return "Planner"


def construct_graspa_mcp_graph(
    llm: ChatOpenAI,
    planner_prompt: str = planner_prompt,
    executor_prompt: str = executor_prompt,
    analyst_prompt: str = analyst_prompt,
    executor_tools: list = None,
    analysis_tools: list = None,
):
    """Construct the gRASPA MCP map-reduce graph.

    Parameters
    ----------
    llm : ChatOpenAI
        Chat model shared by planner, executors, and analyst.
    planner_prompt : str, optional
        Planner system prompt.
    executor_prompt : str, optional
        Executor system prompt.
    analyst_prompt : str, optional
        Analyst system prompt.
    executor_tools : list, optional
        Tools available to executor subgraphs.
    analysis_tools : list, optional
        Tools available to the analyst node.

    Returns
    -------
    CompiledStateGraph
        Compiled gRASPA MCP graph.
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
    graph_builder.add_node("analyst_tools", ToolNode(analysis_tools))

    # -- Edges --
    graph_builder.set_entry_point("Planner")

    # Conditional Edge: Planner -> [Send(executor)...]
    graph_builder.add_conditional_edges(
        "Planner",
        unified_planner_router,
        [
            "insight_analyst",
            "executor_subgraph",
            END,
        ],
    )
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
