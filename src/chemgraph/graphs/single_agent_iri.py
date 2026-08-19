"""single_agent_iri: single-agent LangGraph bound to ALCF IRI tools.

Same shape as graphs/graspa_agent.py -- one LLM node, one ToolNode,
route-tools edge. Two tool sets are shipped:

  ALCF_IRI_FLAT_TOOLS     (43 direct wrappers, default -- higher judge score)
  ALCF_IRI_CATEGORY_TOOLS (7 category tools + discovery, smaller schema)

Pick either at construction time via ``tools=...``. If a matching system
prompt isn't passed, the graph auto-selects one based on which tool set
is bound (category -> alcf_iri_prompt, flat -> alcf_iri_flat_prompt).
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from chemgraph.tools.alcf_iri_tools import ALCF_IRI_CATEGORY_TOOLS
from chemgraph.tools.alcf_iri_flat_tools import ALCF_IRI_FLAT_TOOLS
from chemgraph.prompt.alcf_iri_prompt import alcf_iri_prompt, alcf_iri_flat_prompt
from chemgraph.prompt.single_agent_prompt import formatter_prompt
from chemgraph.schemas.agent_response import ResponseFormatter
from chemgraph.state.state import State
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_DEFAULT_CHECKPOINTER = object()


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
        tools = ALCF_IRI_FLAT_TOOLS
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


def _default_prompt_for(tools) -> str:
    """Pick the system prompt that matches the bound tool set.

    Category tools use the dispatcher/discovery protocol; flat tools are
    called directly. The wrong prompt hurts quality because it argues
    against the tool structure the model sees.
    """
    if tools is ALCF_IRI_CATEGORY_TOOLS:
        return alcf_iri_prompt
    # Default (flat) and any custom mix
    return alcf_iri_flat_prompt


def construct_iri_graph(
    llm: ChatOpenAI,
    system_prompt: str | None = None,
    structured_output: bool = False,
    formatter_prompt: str = formatter_prompt,
    tools: list | None = None,
    checkpointer=_DEFAULT_CHECKPOINTER,
):
    """Construct the single-agent IRI graph.

    Parameters
    ----------
    tools : list, optional
        Tool list to bind. Defaults to ``ALCF_IRI_FLAT_TOOLS`` (winner on
        our judge-scored eval). Pass ``ALCF_IRI_CATEGORY_TOOLS`` for the
        smaller-schema-surface discovery variant.
    system_prompt : str, optional
        System prompt. If omitted, auto-selected based on ``tools``:
        category -> ``alcf_iri_prompt``, flat/other -> ``alcf_iri_flat_prompt``.
    checkpointer : optional
        LangGraph checkpointer used to compile the graph. When omitted, a new
        ``MemorySaver`` preserves standalone behavior. Pass ``None`` when
        embedding this graph so it inherits the parent checkpointer.
    """
    logger.info("Constructing single_agent_iri graph")
    if checkpointer is _DEFAULT_CHECKPOINTER:
        checkpointer = MemorySaver()
    if tools is None:
        tools = ALCF_IRI_FLAT_TOOLS
    if system_prompt is None:
        system_prompt = _default_prompt_for(tools)
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
