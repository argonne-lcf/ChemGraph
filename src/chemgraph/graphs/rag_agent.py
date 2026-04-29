"""LangGraph workflow for the RAG (Retrieval-Augmented Generation) agent.

This graph combines document retrieval tools (load_document,
query_knowledge_base) with the standard chemistry tools so the agent
can answer questions grounded in user-provided text documents *and*
run molecular simulations when needed.

Graph structure
---------------

    START
      |
      v
  RAGAgent  <-------+
      |              |
     (route)         |
     / \\             |
    v    v           |
  tools  done-->END  |
    |                |
    +----------------+

The agent loops through a ReAct cycle: it can call any combination of
RAG tools and chemistry tools, inspect the results, and decide whether
to call more tools or produce a final answer.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from chemgraph.tools.rag_tools import load_document, query_knowledge_base
from chemgraph.tools.ase_tools import (
    run_ase,
    save_atomsdata_to_file,
    file_to_atomsdata,
)
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_coordinate_file,
)
from chemgraph.tools.generic_tools import calculator
from chemgraph.prompt.rag_prompt import rag_agent_prompt
from chemgraph.state.state import State
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers (reuse the repeated-tool-call detection from single_agent)
# ---------------------------------------------------------------------------
def _tool_call_signature(tool_calls) -> tuple:
    """Create a comparable signature for a list of tool calls."""
    signature = []
    for call in tool_calls or []:
        name = call.get("name") if isinstance(call, dict) else None
        args = call.get("args", {}) if isinstance(call, dict) else {}
        if isinstance(args, dict):
            args_sig = tuple(sorted(args.items()))
        else:
            args_sig = str(args)
        signature.append((name, args_sig))
    return tuple(signature)


def _is_repeated_tool_cycle(messages) -> bool:
    """Detect if the most recent AI tool-call set repeats the previous one."""
    ai_with_calls = [
        m
        for m in messages
        if hasattr(m, "tool_calls") and getattr(m, "tool_calls", None)
    ]
    if len(ai_with_calls) < 2:
        return False
    last = _tool_call_signature(ai_with_calls[-1].tool_calls)
    prev = _tool_call_signature(ai_with_calls[-2].tool_calls)
    return bool(last) and last == prev


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_tools(state: State):
    """Route to 'tools' if the last message has tool calls, else 'done'.

    Parameters
    ----------
    state : State
        Current graph state.

    Returns
    -------
    str
        ``"tools"`` or ``"done"``.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if not isinstance(state, list) and _is_repeated_tool_cycle(messages):
            return "done"
        return "tools"
    return "done"


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------
def RAGAgent(state: State, llm, system_prompt: str, tools=None):
    """LLM node that can retrieve from documents and run chemistry tools.

    Parameters
    ----------
    state : State
        Current graph state with messages.
    llm : BaseChatModel
        The bound language model.
    system_prompt : str
        System prompt guiding the agent's behaviour.
    tools : list, optional
        Tools available to the agent. Uses the default RAG + chemistry
        tool set when ``None``.

    Returns
    -------
    dict
        Updated state with the LLM's response appended to messages.
    """
    if tools is None:
        tools = _default_tools()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


# ---------------------------------------------------------------------------
# Default tool set
# ---------------------------------------------------------------------------
def _default_tools():
    """Return the combined RAG + chemistry tool list."""
    return [
        # RAG tools
        load_document,
        query_knowledge_base,
        # Chemistry tools
        file_to_atomsdata,
        smiles_to_coordinate_file,
        run_ase,
        molecule_name_to_smiles,
        save_atomsdata_to_file,
        calculator,
    ]


# ---------------------------------------------------------------------------
# Graph constructor
# ---------------------------------------------------------------------------
def construct_rag_agent_graph(
    llm,
    system_prompt: str = rag_agent_prompt,
    tools: list = None,
):
    """Construct a RAG agent graph with document retrieval and chemistry tools.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to power the agent.
    system_prompt : str, optional
        System prompt for the RAG agent, by default ``rag_agent_prompt``.
    tools : list, optional
        Custom tool list. When ``None`` the default RAG + chemistry
        tools are used.

    Returns
    -------
    CompiledStateGraph
        The compiled LangGraph workflow ready for execution.
    """
    try:
        logger.info("Constructing RAG agent graph")
        checkpointer = MemorySaver()

        if tools is None:
            tools = _default_tools()

        tool_node = ToolNode(tools=tools)
        graph_builder = StateGraph(State)

        # Nodes
        graph_builder.add_node(
            "RAGAgent",
            lambda state: RAGAgent(
                state, llm, system_prompt=system_prompt, tools=tools
            ),
        )
        graph_builder.add_node("tools", tool_node)

        # Edges
        graph_builder.add_edge(START, "RAGAgent")
        graph_builder.add_conditional_edges(
            "RAGAgent",
            route_tools,
            {"tools": "tools", "done": END},
        )
        graph_builder.add_edge("tools", "RAGAgent")

        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("RAG agent graph construction completed")
        return graph

    except Exception as e:
        logger.error(f"Error constructing RAG agent graph: {e}")
        raise
