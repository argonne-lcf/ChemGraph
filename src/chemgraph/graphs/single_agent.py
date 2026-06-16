import json
from collections.abc import Collection

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from chemgraph.tools.ase_tools import (
    run_ase,
    extract_output_json,
)
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_coordinate_file,
)
from chemgraph.tools.report_tools import generate_html
from chemgraph.tools.generic_tools import calculator, ask_human
from chemgraph.prompt.single_agent_prompt import (
    single_agent_prompt,
    formatter_prompt,
    report_prompt,
)
from chemgraph.utils.logging_config import setup_logger
from chemgraph.utils.parsing import parse_response_formatter
from chemgraph.state.state import State

logger = setup_logger(__name__)


def _tool_call_signature(tool_calls) -> tuple:
    """Create a comparable signature for a list of tool calls.

    Parameters
    ----------
    tool_calls : list
        Tool-call dictionaries from an AI message.

    Returns
    -------
    tuple
        Deterministic signature of tool names and arguments.
    """
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
    """Detect if the most recent AI tool-call set repeats the previous one.

    Parameters
    ----------
    messages : list
        Message history to inspect.

    Returns
    -------
    bool
        ``True`` when the last two AI tool-call sets are identical.
    """
    ai_with_calls = []
    for message in messages:
        if hasattr(message, "tool_calls") and getattr(message, "tool_calls", None):
            ai_with_calls.append(message)

    if len(ai_with_calls) < 2:
        return False

    last_calls = _tool_call_signature(ai_with_calls[-1].tool_calls)
    prev_calls = _tool_call_signature(ai_with_calls[-2].tool_calls)
    return bool(last_calls) and last_calls == prev_calls


def _tool_message_name(message):
    """Extract tool name from a message-like object.

    Parameters
    ----------
    message : Any
        Message dictionary or object.

    Returns
    -------
    str or None
        Tool name when present.
    """
    if isinstance(message, dict):
        return message.get("name")
    return getattr(message, "name", None)


def _tool_message_content(message):
    """Extract content text from a message-like object.

    Parameters
    ----------
    message : Any
        Message dictionary or object.

    Returns
    -------
    Any
        Message content, or an empty string when unavailable.
    """
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _message_tool_calls(message) -> list:
    """Extract tool calls from a message-like object."""
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    return calls if isinstance(calls, list) else []


def _state_messages(state: State):
    """Extract messages from a LangGraph state or message list."""
    if isinstance(state, list):
        return state
    if messages := state.get("messages", []):
        return messages
    raise ValueError(f"No messages found in input state to tool_edge: {state}")


def _tool_result_names_after_latest_ai_tool_call(messages) -> set[str]:
    """Return tool-result names appended after the latest AI tool-call message."""
    names: set[str] = set()
    for message in reversed(messages):
        if _message_tool_calls(message):
            return names
        name = _tool_message_name(message)
        if name:
            names.add(str(name))
    return names


def _is_successful_report_message(message) -> bool:
    """Return True when a message indicates successful report generation.

    Parameters
    ----------
    message : Any
        Tool message dictionary or object.

    Returns
    -------
    bool
        ``True`` for non-error ``generate_html`` tool output.
    """
    if _tool_message_name(message) != "generate_html":
        return False

    content = _tool_message_content(message)
    content_text = str(content).strip().lower() if content is not None else ""
    if not content_text:
        return False

    # ToolNode formats failures as "Error: ..."; treat only non-error output as success.
    return not content_text.startswith("error")


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
    messages = _state_messages(state)
    ai_message = messages[-1]
    if _message_tool_calls(ai_message):
        if not isinstance(state, list) and _is_repeated_tool_cycle(messages):
            return "done"
        return "tools"
    return "done"


def route_after_tools(
    state: State,
    terminal_tool_names: Collection[str] = (),
):
    """Stop the graph after terminal tools; otherwise continue to the LLM."""
    if not terminal_tool_names:
        return "continue"
    executed_names = _tool_result_names_after_latest_ai_tool_call(
        _state_messages(state),
    )
    terminal_names = {str(name) for name in terminal_tool_names}
    return "done" if executed_names & terminal_names else "continue"


def route_report_tools(state: State):
    """Route report tool execution and stop if a report was already generated.

    Parameters
    ----------
    state : State
        Current graph state or message list.

    Returns
    -------
    str
        ``"tools"`` when ``generate_html`` should run, otherwise ``"done"``.
    """
    if isinstance(state, list):
        messages = state
        ai_message = state[-1] if state else None
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    tool_calls = _message_tool_calls(ai_message)
    if not tool_calls:
        return "done"

    # Only allow known report tool calls to reach ToolNode.
    valid_report_tools = {"generate_html"}
    requested_tools = {
        call.get("name")
        for call in tool_calls
        if isinstance(call, dict)
    }
    if not requested_tools or not requested_tools.issubset(valid_report_tools):
        return "done"

    report_generated = any(
        _is_successful_report_message(message) for message in messages
    )
    return "done" if report_generated else "tools"


def route_after_report_tools(state: State):
    """After report tool execution, stop on success or retry on failure.

    Parameters
    ----------
    state : State
        Current graph state or message list after report tool execution.

    Returns
    -------
    str
        ``"done"`` after a successful report message, otherwise ``"retry"``.
    """
    if isinstance(state, list):
        messages = state
    elif messages := state.get("messages", []):
        pass
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    return "done" if _is_successful_report_message(messages[-1]) else "retry"


def ChemGraphAgent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str,
    tools=None,
    human_supervised: bool = False,
):
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
    human_supervised : bool, optional
        Whether to include the ``ask_human`` tool, by default False

    Returns
    -------
    dict
        Updated state containing the LLM's response
    """

    # Load default tools if no tool is specified.
    if tools is None:
        tools = [
            smiles_to_coordinate_file,
            run_ase,
            molecule_name_to_smiles,
            extract_output_json,
            calculator,
        ]
        if human_supervised:
            tools.append(ask_human)
    elif human_supervised and ask_human not in tools:
        # Ensure ask_human is available when custom tools are provided
        # and human supervision is enabled.
        tools = list(tools) + [ask_human]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def ResponseAgent(
    state: State,
    llm: ChatOpenAI,
    formatter_prompt: str,
    max_retries: int = 1,
):
    """An LLM agent responsible for formatting final message.

    When the LLM response cannot be parsed into a valid
    :class:`ResponseFormatter`, the agent retries the LLM call up to
    ``max_retries`` times, sending the parse error back to the model so
    it can correct its output.

    If all attempts fail, an empty ``ResponseFormatter`` is returned
    with a ``_parse_error`` key in the serialised JSON so that
    downstream evaluation can detect the failure.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps.
    llm : ChatOpenAI
        The language model to use for formatting.
    formatter_prompt : str
        The prompt to guide the LLM's formatting behaviour.
    max_retries : int, optional
        Maximum number of retry attempts on parse failure (default 1).

    Returns
    -------
    dict
        Updated state containing the formatted response.
    """
    messages = [
        {"role": "system", "content": formatter_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    raw_response = llm.invoke(messages).content
    formatter, parse_error = parse_response_formatter(raw_response)

    # Retry loop: re-invoke the LLM with the error feedback.
    retries = 0
    while parse_error is not None and retries < max_retries:
        retries += 1
        logger.warning(
            "ResponseAgent: parse attempt %d failed (%s); retrying LLM.",
            retries,
            parse_error,
        )
        retry_messages = [
            {"role": "system", "content": formatter_prompt},
            {"role": "user", "content": f"{state['messages']}"},
            {
                "role": "assistant",
                "content": raw_response,
            },
            {
                "role": "user",
                "content": (
                    f"Error: {parse_error}\n\n"
                    "Your previous response could not be parsed. "
                    "Please output ONLY a valid JSON object matching the "
                    "ResponseFormatter schema. Do not include any text, "
                    "markdown fences, or explanation outside the JSON object."
                ),
            },
        ]
        raw_response = llm.invoke(retry_messages).content
        formatter, parse_error = parse_response_formatter(raw_response)

    # Serialise to JSON, injecting ``_parse_error`` when parsing failed.
    result = json.loads(formatter.model_dump_json())
    if parse_error is not None:
        logger.error(
            "ResponseAgent: all %d retries exhausted; returning empty "
            "ResponseFormatter with _parse_error.",
            max_retries,
        )
        result["_parse_error"] = parse_error
    response = json.dumps(result)
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
    llm_with_tools = llm.bind_tools(
        tools=tools,
        tool_choice="generate_html",
        parallel_tool_calls=False,
    )
    return {"messages": [llm_with_tools.invoke(messages)]}


def construct_single_agent_graph(
    llm: ChatOpenAI,
    system_prompt: str = single_agent_prompt,
    structured_output: bool = False,
    formatter_prompt: str = formatter_prompt,
    generate_report: bool = False,
    report_prompt: str = report_prompt,
    tools: list = None,
    max_retries: int = 1,
    human_supervised: bool = False,
    terminal_tool_names: Collection[str] = (),
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
    tools : list, optional
        The list of tools for the main agent, by default None
    max_retries : int, optional
        Maximum number of LLM retry attempts when the ResponseAgent
        fails to parse the formatter output, by default 1
    human_supervised : bool, optional
        Whether to include the ``ask_human`` tool so the agent can
        pause and request human input, by default False
    terminal_tool_names : Collection[str], optional
        Tool names that should terminate the graph after successful tool
        execution instead of routing back to the LLM, by default empty.

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
                smiles_to_coordinate_file,
                molecule_name_to_smiles,
                run_ase,
                extract_output_json,
                calculator,
            ]
            if human_supervised:
                tools.append(ask_human)
        elif human_supervised and ask_human not in tools:
            # Ensure ask_human is available when custom tools are provided
            # and human supervision is enabled.
            tools = list(tools) + [ask_human]
        tool_node = ToolNode(tools=tools)
        graph_builder = StateGraph(State)

        if not structured_output:
            graph_builder.add_node(
                "ChemGraphAgent",
                lambda state: ChemGraphAgent(
                    state,
                    llm,
                    system_prompt=system_prompt,
                    tools=tools,
                    human_supervised=human_supervised,
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
                graph_builder.add_conditional_edges(
                    "tools",
                    lambda state: route_after_tools(state, terminal_tool_names),
                    {"continue": "ChemGraphAgent", "done": END},
                )
                graph_builder.add_conditional_edges(
                    "ReportAgent",
                    route_report_tools,
                    {"tools": "report_tools", "done": END},
                )
                graph_builder.add_conditional_edges(
                    "report_tools",
                    route_after_report_tools,
                    {"retry": "ReportAgent", "done": END},
                )
            else:
                graph_builder.add_conditional_edges(
                    "ChemGraphAgent",
                    route_tools,
                    {"tools": "tools", "done": END},
                )
                graph_builder.add_conditional_edges(
                    "tools",
                    lambda state: route_after_tools(state, terminal_tool_names),
                    {"continue": "ChemGraphAgent", "done": END},
                )

            graph = graph_builder.compile(checkpointer=checkpointer)
            logger.info("Graph construction completed")
            return graph
        else:
            graph_builder.add_node(
                "ChemGraphAgent",
                lambda state: ChemGraphAgent(
                    state,
                    llm,
                    system_prompt=system_prompt,
                    tools=tools,
                    human_supervised=human_supervised,
                ),
            )
            graph_builder.add_node("tools", tool_node)
            graph_builder.add_node(
                "ResponseAgent",
                lambda state: ResponseAgent(
                    state,
                    llm,
                    formatter_prompt=formatter_prompt,
                    max_retries=max_retries,
                ),
            )
            graph_builder.add_conditional_edges(
                "ChemGraphAgent",
                route_tools,
                {"tools": "tools", "done": "ResponseAgent"},
            )
            graph_builder.add_conditional_edges(
                "tools",
                lambda state: route_after_tools(state, terminal_tool_names),
                {"continue": "ChemGraphAgent", "done": END},
            )
            graph_builder.add_edge(START, "ChemGraphAgent")
            graph_builder.add_edge("ResponseAgent", END)

            graph = graph_builder.compile(checkpointer=checkpointer)
            logger.info("Graph construction completed")
            return graph

    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
