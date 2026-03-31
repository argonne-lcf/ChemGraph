import os

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_coordinate_file,
)
from chemgraph.tools.ase_tools import run_ase
from chemgraph.tools.xanes_tools import (
    run_xanes, 
    fetch_xanes_data, 
    plot_xanes_data,)
from chemgraph.schemas.agent_response import ResponseFormatter
from chemgraph.prompt.xanes_prompt import (
    xanes_single_agent_prompt,
    xanes_formatter_prompt,
)
from chemgraph.utils.logging_config import setup_logger
from chemgraph.state.state import State

logger = setup_logger(__name__)


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
    """Detect if the most recent AI tool-call set repeats the previous AI tool-call set."""
    ai_with_calls = []
    for message in messages:
        if hasattr(message, "tool_calls") and getattr(message, "tool_calls", None):
            ai_with_calls.append(message)

    if len(ai_with_calls) < 2:
        return False

    last_calls = _tool_call_signature(ai_with_calls[-1].tool_calls)
    prev_calls = _tool_call_signature(ai_with_calls[-2].tool_calls)
    return bool(last_calls) and last_calls == prev_calls


def route_tools(state: State):
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
        if not isinstance(state, list) and _is_repeated_tool_cycle(messages):
            return "done"
        return "tools"
    return "done"


def XANESAgent(state: State, llm: ChatOpenAI, system_prompt: str, tools=None):
    """LLM node for XANES workflows that processes messages and decides next actions.

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
    if tools is None:
        tools = [
            molecule_name_to_smiles,
            smiles_to_coordinate_file,
            run_ase,
            run_xanes,
            fetch_xanes_data,
            plot_xanes_data,
        ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def ResponseAgent(state: State, llm: ChatOpenAI, formatter_prompt: str):
    """An LLM agent responsible for formatting final message.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps
    llm : ChatOpenAI
        The language model to use for formatting
    formatter_prompt : str
        The prompt to guide the LLM's formatting behavior

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


def construct_single_agent_xanes_graph(
    llm: ChatOpenAI,
    system_prompt: str = xanes_single_agent_prompt,
    structured_output: bool = False,
    formatter_prompt: str = xanes_formatter_prompt,
    tools: list = None,
):
    """Construct a single-agent graph for XANES/FDMNES workflows.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use for the graph
    system_prompt : str, optional
        The system prompt to guide the LLM's behavior,
        by default xanes_single_agent_prompt
    structured_output : bool, optional
        Whether to use structured output, by default False
    formatter_prompt : str, optional
        The prompt to guide the LLM's formatting behavior,
        by default xanes_formatter_prompt
    tools : list, optional
        The list of tools for the main agent, by default None

    Returns
    -------
    StateGraph
        The constructed single agent XANES graph
    """
    try:
        logger.info("Constructing single agent XANES graph")

        if not os.environ.get("MP_API_KEY"):
            logger.warning(
                "MP_API_KEY environment variable is not set. "
                "The fetch_xanes_data tool will require an API key "
                "to be passed explicitly."
            )
        if not os.environ.get("FDMNES_EXE"):
            logger.warning(
                "FDMNES_EXE environment variable is not set. "
                "The run_xanes tool will not work without the FDMNES executable."
            )

        checkpointer = MemorySaver()
        if tools is None:
            tools = [
                molecule_name_to_smiles,
                smiles_to_coordinate_file,
                run_ase,
                run_xanes,
                fetch_xanes_data,
                plot_xanes_data,
            ]
        tool_node = ToolNode(tools=tools)
        graph_builder = StateGraph(State)

        if not structured_output:
            graph_builder.add_node(
                "XANESAgent",
                lambda state: XANESAgent(
                    state, llm, system_prompt=system_prompt, tools=tools
                ),
            )
            graph_builder.add_node("tools", tool_node)
            graph_builder.add_edge(START, "XANESAgent")
            graph_builder.add_conditional_edges(
                "XANESAgent",
                route_tools,
                {"tools": "tools", "done": END},
            )
            graph_builder.add_edge("tools", "XANESAgent")
            graph_builder.add_edge("XANESAgent", END)

            graph = graph_builder.compile(checkpointer=checkpointer)
            logger.info("XANES graph construction completed")
            return graph
        else:
            graph_builder.add_node(
                "XANESAgent",
                lambda state: XANESAgent(
                    state, llm, system_prompt=system_prompt, tools=tools
                ),
            )
            graph_builder.add_node("tools", tool_node)
            graph_builder.add_node(
                "ResponseAgent",
                lambda state: ResponseAgent(
                    state, llm, formatter_prompt=formatter_prompt
                ),
            )
            graph_builder.add_conditional_edges(
                "XANESAgent",
                route_tools,
                {"tools": "tools", "done": "ResponseAgent"},
            )
            graph_builder.add_edge("tools", "XANESAgent")
            graph_builder.add_edge(START, "XANESAgent")
            graph_builder.add_edge("ResponseAgent", END)

            graph = graph_builder.compile(checkpointer=checkpointer)
            logger.info("XANES graph construction completed")
            return graph

    except Exception as e:
        logger.error(f"Error constructing XANES graph: {str(e)}")
        raise
