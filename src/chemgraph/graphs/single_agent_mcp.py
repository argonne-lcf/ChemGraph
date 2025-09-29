from typing import List, Any

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from chemgraph.prompt.single_agent_prompt import (
    single_agent_prompt,
)
from chemgraph.utils.logging_config import setup_logger
from chemgraph.state.state import State
from langgraph.prebuilt import ToolNode

logger = setup_logger(__name__)


def route_tools(state: State) -> str:
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps

    Returns
    -------
    str
        Either 'tools' or 'done' based on the state conditions
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


def ChemGraphAgent(state: State, llm: ChatOpenAI, system_prompt: str, tools=None):
    """LLM node that processes messages and decides next actions.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps
    llm : ChatOpenAI
        The language model to use for processing
    system_prompt : str
        The system prompt to guide the LLM's behavior
    tools : list, optional
        List of tools available to the agent, by default None

    Returns
    -------
    dict
        Updated state containing the LLM's response
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def construct_single_agent_mcp_graph(
    llm: ChatOpenAI,
    system_prompt: str = single_agent_prompt,
    tools: List[Any] = None,
):
    """Construct a geometry optimization graph.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use for the graph
    system_prompt : str, optional
        The system prompt to guide the LLM's behavior, by default single_agent_prompt
    Returns
    -------
    StateGraph
        The constructed single agent graph
    """
    if not tools:
        raise ValueError("No MCP tools loaded. Ensure MCP servers are configured and reachable.")
    logger.info("Constructing single agent MCP graph (sync)")

    checkpointer = MemorySaver()
    tool_node = ToolNode(tools=tools)
    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "ChemGraphAgent",
        lambda state: ChemGraphAgent(state, llm, system_prompt=system_prompt, tools=tools),
    )
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "ChemGraphAgent")

    graph_builder.add_conditional_edges(
        "ChemGraphAgent",
        route_tools,
        {"tools": "tools", "done": END},
    )
    graph_builder.add_edge("tools", "ChemGraphAgent")
    graph_builder.add_edge("ChemGraphAgent", END)

    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Graph construction completed")
    return graph
