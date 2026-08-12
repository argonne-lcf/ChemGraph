"""single_agent_iri: single-agent LangGraph bound to the six ALCF IRI tools.

Same shape as graphs/graspa_agent.py -- one LLM node, one ToolNode,
route-tools edge. Only difference is the tool list.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from chemgraph.tools.alcf_iri_tools import ALCF_IRI_TOOLS
from chemgraph.prompt.alcf_iri_prompt import alcf_iri_prompt
from chemgraph.prompt.single_agent_prompt import formatter_prompt
from chemgraph.schemas.agent_response import ResponseFormatter
from chemgraph.state.state import State
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


def ChemGraphAgent(state: State, llm: ChatOpenAI, system_prompt: str, tools=None):
    if tools is None:
        tools = ALCF_IRI_TOOLS
    llm_with_tools = llm.bind_tools(tools=tools)
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def ResponseAgent(state: State, llm: ChatOpenAI, formatter_prompt: str):
    messages = [
        {"role": "system", "content": formatter_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_structured_output = llm.with_structured_output(ResponseFormatter)
    response = llm_structured_output.invoke(messages).model_dump_json()
    return {"messages": [response]}


def construct_iri_graph(
    llm: ChatOpenAI,
    system_prompt: str = alcf_iri_prompt,
    structured_output: bool = False,
    formatter_prompt: str = formatter_prompt,
    tools: list = None,
):
    """Construct the single-agent IRI graph."""
    logger.info("Constructing single_agent_iri graph")
    checkpointer = MemorySaver()
    if tools is None:
        tools = ALCF_IRI_TOOLS
    tool_node = ToolNode(tools=tools)
    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "ChemGraphAgent",
        lambda state: ChemGraphAgent(
            state, llm, system_prompt=system_prompt, tools=tools,
        ),
    )
    graph_builder.add_node("tools", tool_node)

    if structured_output:
        graph_builder.add_node(
            "ResponseAgent",
            lambda state: ResponseAgent(state, llm, formatter_prompt=formatter_prompt),
        )
        graph_builder.add_conditional_edges(
            "ChemGraphAgent", route_tools,
            {"tools": "tools", "done": "ResponseAgent"},
        )
        graph_builder.add_edge("ResponseAgent", END)
    else:
        graph_builder.add_conditional_edges(
            "ChemGraphAgent", route_tools,
            {"tools": "tools", "done": END},
        )

    graph_builder.add_edge("tools", "ChemGraphAgent")
    graph_builder.add_edge(START, "ChemGraphAgent")

    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info("single_agent_iri graph construction completed")
    return graph
