from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from chemgraph.tools.ase_tools import (
    run_ase,
    save_atomsdata_to_file,
    file_to_atomsdata,
)
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_coordinate_file,
)
from chemgraph.tools.report_tools import generate_html
from chemgraph.tools.generic_tools import calculator
from chemgraph.schemas.agent_response import ResponseFormatter
from chemgraph.prompt.single_agent_prompt import (
    single_agent_prompt,
    formatter_prompt,
    report_prompt,
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
        # Normalize args for deterministic comparisons across repeated cycles.
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


def route_report_tools(state: State):
    """Route report tool execution and stop if a report was already generated."""
    if isinstance(state, list):
        messages = state
        ai_message = state[-1] if state else None
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if not (hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0):
        return "done"

    report_exists = any(
        isinstance(message, dict) and message.get("name") == "generate_html"
        for message in messages
    )
    return "done" if report_exists else "tools"


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

    # Load default tools if no tool is specified.
    if tools is None:
        tools = [
            file_to_atomsdata,
            smiles_to_coordinate_file,
            run_ase,
            molecule_name_to_smiles,
            save_atomsdata_to_file,
            calculator,
        ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def ResponseAgent(state: State, llm: ChatOpenAI, formatter_prompt: str):
    """An LLM agent responsible for formatting final message

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


def ReportAgent(
    state: State, llm: ChatOpenAI, system_prompt: str, tools=[generate_html]
):
    """LLM node that generates a report from the messages.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps
    llm : ChatOpenAI
        The language model to use for processing
    system_prompt : str
        The system prompt to guide the LLM's behavior
    tools : list, optional
        List of tools available to the agent, by default [generate_html]

    Returns
    -------
    dict
        Updated state containing the LLM's response
    """

    # Load default tools if no tool is specified.
    if tools is None:
        tools = [
            generate_html,
        ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def construct_single_agent_graph(
    llm: ChatOpenAI,
    system_prompt: str = single_agent_prompt,
    structured_output: bool = False,
    formatter_prompt: str = formatter_prompt,
    generate_report: bool = False,
    report_prompt: str = report_prompt,
    tools: list = None,
):
    """Construct a geometry optimization graph.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use for the graph
    system_prompt : str, optional
        The system prompt to guide the LLM's behavior, by default single_agent_prompt
    structured_output : bool, optional
        Whether to use structured output, by default False
    formatter_prompt : str, optional
        The prompt to guide the LLM's formatting behavior, by default formatter_prompt
    generate_report: bool, optional
        Whether to generate a report, by default False
    report_prompt: str, optional
        The prompt to guide the LLM's report generation behavior, by default report_prompt
    tool: list, optional
        The list of tools for the main agent, by default None
    Returns
    -------
    StateGraph
        The constructed single agent graph
    """
    try:
        logger.info("Constructing single agent graph")
        checkpointer = MemorySaver()
        if tools is None:
            tools = [
                file_to_atomsdata,
                smiles_to_coordinate_file,
                run_ase,
                molecule_name_to_smiles,
                save_atomsdata_to_file,
                calculator,
            ]
        tool_node = ToolNode(tools=tools)
        graph_builder = StateGraph(State)

        if not structured_output:
            graph_builder.add_node(
                "ChemGraphAgent",
                lambda state: ChemGraphAgent(
                    state, llm, system_prompt=system_prompt, tools=tools
                ),
            )
            graph_builder.add_node("tools", tool_node)
            graph_builder.add_edge(START, "ChemGraphAgent")

            if generate_report:
                tool_node_report = ToolNode(tools=[generate_html])
                graph_builder.add_node("report_tools", tool_node_report)

                graph_builder.add_node(
                    "ReportAgent",
                    lambda state: ReportAgent(
                        state, llm, system_prompt=report_prompt, tools=[generate_html]
                    ),
                )
                graph_builder.add_conditional_edges(
                    "ChemGraphAgent",
                    route_tools,
                    {"tools": "tools", "done": "ReportAgent"},
                )
                graph_builder.add_edge("tools", "ChemGraphAgent")
                graph_builder.add_conditional_edges(
                    "ReportAgent",
                    route_tools,
                    {"tools": "report_tools", "done": END},
                )
                graph_builder.add_edge("report_tools", "ReportAgent")
            else:
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
        else:
            graph_builder.add_node(
                "ChemGraphAgent",
                lambda state: ChemGraphAgent(
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
                "ChemGraphAgent",
                route_tools,
                {"tools": "tools", "done": "ResponseAgent"},
            )
            graph_builder.add_edge("tools", "ChemGraphAgent")
            graph_builder.add_edge(START, "ChemGraphAgent")
            graph_builder.add_edge("ResponseAgent", END)

            graph = graph_builder.compile(checkpointer=checkpointer)
            logger.info("Graph construction completed")
            return graph

    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
