"""Multi-agent workflow using the LangGraph Send() (map-reduce) pattern.

Architecture
------------
Main graph (``PlannerState``)::

    Planner --condition--> Send(executor_subgraph, task1..N) --> Planner
                       |-> ResponseAgent --> END   (when structured_output)
                       |-> END                     (when FINISH, no formatting)

Executor subgraph (``ExecutorState``)::

    executor_agent --> ToolNode --> executor_agent  (ReAct loop)
                   |-> finalize --> END             (no more tool calls)
"""

import json
from typing import Any, Union
from functools import partial

from pydantic import BaseModel
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send, interrupt

from chemgraph.utils.logging_config import setup_logger
from chemgraph.utils.parsing import extract_json_block, parse_response_formatter
from chemgraph.state.multi_agent_state import ExecutorState, PlannerState
from chemgraph.schemas.multi_agent_response import PlannerResponse
from chemgraph.prompt.multi_agent_prompt import (
    planner_prompt as default_planner_prompt,
    executor_prompt as default_executor_prompt,
    formatter_multi_prompt as default_formatter_prompt,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    else:
        return obj


def sanitize_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure tool_call['args'] contains only JSON-serializable data.

    After LangChain's ToolNode validates tool-call arguments against
    Pydantic schemas (e.g. ``ASEInputSchema``), nested calculator dicts
    may be replaced by live Pydantic objects (e.g. ``MaceCalc``).  When
    these messages are later re-sent to the LLM, LangChain serialises
    ``tool_call['args']`` with ``json.dumps`` — which raises
    ``TypeError`` for Pydantic instances.

    This function walks every ``AIMessage.tool_calls`` entry and
    recursively converts Pydantic models back to plain dicts.
    """
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            new_tool_calls = []
            for tc in m.tool_calls:
                tc = dict(tc)
                tc["args"] = _to_jsonable(tc.get("args"))
                new_tool_calls.append(tc)
            m.tool_calls = new_tool_calls
    return messages


# ---------------------------------------------------------------------------
# Planner helpers
# ---------------------------------------------------------------------------


def _parse_planner_response(
    raw_text: str,
) -> tuple[PlannerResponse | None, str | None]:
    """Parse raw LLM text into a :class:`PlannerResponse`.

    Returns ``(parsed_response, None)`` on success,
    or ``(None, error_msg)`` on failure.
    """
    # 1. Direct validation
    try:
        return PlannerResponse.model_validate_json(raw_text.strip()), None
    except Exception:
        pass

    # 2. Extract JSON block (handles ```json ... ``` or bare {})
    extracted = extract_json_block(raw_text)
    if extracted:
        try:
            return PlannerResponse.model_validate_json(extracted), None
        except Exception:
            pass
        try:
            return PlannerResponse.model_validate(json.loads(extracted)), None
        except Exception:
            pass

    # 3. All attempts failed
    return None, f"Could not parse planner response from: {raw_text[:200]}"


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------


def planner_agent(
    state: PlannerState,
    llm: ChatOpenAI,
    system_prompt: str,
    max_retries: int = 1,
):
    """Planner that decomposes tasks and routes the workflow.

    On the first invocation it sees only the user query in ``messages``.
    On subsequent invocations it also sees ``executor_results`` from
    completed executor subgraphs and can decide to re-plan or finish.

    The LLM is prompted to return a JSON object matching the
    ``PlannerResponse`` schema.  If parsing fails, the LLM is retried
    up to ``max_retries`` times with error feedback.
    """
    executor_outputs = state.get("executor_results", [])
    content_block = f"Current Conversation History: {state['messages']}"
    if executor_outputs:
        results_text = "\n".join(
            m.content if hasattr(m, "content") else str(m) for m in executor_outputs
        )
        content_block += (
            f"\n\n### UPDATED: Results from Executor Tasks ###\n{results_text}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_block},
    ]

    raw_response = llm.invoke(messages).content
    response_obj, parse_error = _parse_planner_response(raw_response)

    retries = 0
    while response_obj is None and retries < max_retries:
        retries += 1
        logger.warning(
            "Planner: parse attempt %d failed (%s); retrying.",
            retries,
            parse_error,
        )
        retry_messages = messages + [
            {"role": "assistant", "content": raw_response},
            {
                "role": "user",
                "content": (
                    f"Error: {parse_error}\n\n"
                    "Your previous response could not be parsed. "
                    "Please output ONLY a valid JSON object matching the "
                    "required format. No markdown fences, no text outside "
                    "the JSON."
                ),
            },
        ]
        raw_response = llm.invoke(retry_messages).content
        response_obj, parse_error = _parse_planner_response(raw_response)

    if response_obj is None:
        raise ValueError(
            f"Planner failed to produce valid JSON after "
            f"{max_retries} retries: {parse_error}"
        )

    logger.info("PLANNER: %s", response_obj.model_dump_json())
    current_iterations = state.get("planner_iterations", 0)
    result = {
        "messages": [AIMessage(content=response_obj.thought_process)],
        "next_step": response_obj.next_step,
        "tasks": response_obj.tasks if response_obj.tasks else [],
        "planner_iterations": current_iterations + 1,
    }
    if response_obj.next_step == "ask_human" and response_obj.clarification:
        result["clarification"] = response_obj.clarification
    return result


# ---------------------------------------------------------------------------
# Human review node (interrupt for human-in-the-loop)
# ---------------------------------------------------------------------------


def human_review_node(state: PlannerState):
    """Pause the graph and ask the human for clarification.

    This node calls ``interrupt()`` with the planner's clarification
    question. Execution halts until a human provides a response via
    ``Command(resume=...)``.  The human's answer is injected back into
    the conversation as an ``AIMessage`` summarising what was asked and
    what the human replied, then control returns to the Planner.
    """
    question = state.get("clarification", "Could you please provide more details?")
    logger.info("HUMAN_REVIEW: interrupting with question: %s", question)

    human_response = interrupt({"question": question})

    # Normalise the response to a plain string.
    if isinstance(human_response, dict):
        answer = human_response.get(
            "answer", human_response.get("response", str(human_response))
        )
    else:
        answer = str(human_response)

    logger.info("HUMAN_REVIEW: received response: %s", answer)
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Human clarification received.\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Planner router (conditional edge)
# ---------------------------------------------------------------------------


def unified_planner_router(
    state: PlannerState,
    structured_output: bool = False,
    max_planner_iterations: int = 3,
) -> Union[str, list[Send]]:
    """Route based on the planner's ``next_step`` decision.

    * ``executor_subgraph`` -- fan-out tasks via ``Send()``
    * ``ask_human`` -- pause for human clarification via ``human_review``
    * ``FINISH`` -- go to ``ResponseAgent`` (if structured_output) or ``END``

    A cycle guard forces ``FINISH`` when the planner has dispatched
    executors ``max_planner_iterations`` times to prevent infinite loops.
    """
    next_step = state.get("next_step")
    iterations = state.get("planner_iterations", 0)

    if next_step == "ask_human":
        return "human_review"

    if next_step == "executor_subgraph":
        if iterations > max_planner_iterations:
            logger.warning(
                "Planner exceeded max iterations (%d); forcing FINISH.",
                max_planner_iterations,
            )
            if structured_output:
                return "ResponseAgent"
            return END

        tasks = state.get("tasks", [])
        return [
            Send(
                "executor_subgraph",
                {
                    "executor_id": f"worker_{getattr(t, 'task_index', i + 1)}",
                    "messages": [getattr(t, "prompt", str(t))],
                },
            )
            for i, t in enumerate(tasks)
        ]

    # FINISH
    if structured_output:
        return "ResponseAgent"
    return END


# ---------------------------------------------------------------------------
# Executor subgraph nodes
# ---------------------------------------------------------------------------


async def executor_model_node(
    state: ExecutorState,
    llm: ChatOpenAI,
    system_prompt: str,
    tools: list,
):
    """ReAct reasoning step inside an executor subgraph.

    Reads its own ``messages`` history, calls the LLM with bound tools,
    and returns the response.
    """
    sanitized = sanitize_tool_calls(list(state["messages"]))
    messages = [{"role": "system", "content": system_prompt}] + sanitized

    # Flatten MCP/LangChain content blocks to plain text for ChatOpenAI
    for m in messages:
        content = (
            m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        )
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

    logger.debug("Executor response: %s", response)
    return {"messages": [response]}


def route_executor(state: ExecutorState):
    """Standard ReAct routing: tool calls -> ``tools``, else -> ``done``."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "done"


def format_executor_output(state: ExecutorState) -> dict:
    """Bridge: convert local ``ExecutorState`` into a ``PlannerState`` update.

    Writes the executor's final answer into ``executor_results`` and
    its full message history into ``executor_logs`` so the planner can
    inspect them on the next iteration.
    """
    executor_id = state["executor_id"]
    final_message = state["messages"][-1].content
    full_history = state["messages"]

    return {
        "executor_results": [f"[{executor_id}] Result: {final_message}"],
        "executor_logs": {executor_id: full_history},
    }


def construct_executor_subgraph(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
):
    """Build the reusable executor subgraph (Agent -> Tools -> Agent loop).

    The subgraph is compiled and used as a node in the main graph.
    Each ``Send()`` invocation creates an independent copy with its own
    ``ExecutorState``.
    """
    workflow = StateGraph(ExecutorState)
    workflow.add_node(
        "executor_agent",
        partial(
            executor_model_node, llm=llm, system_prompt=system_prompt, tools=tools
        ),
    )
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("finalize", format_executor_output)

    workflow.set_entry_point("executor_agent")
    workflow.add_conditional_edges(
        "executor_agent",
        route_executor,
        {"tools": "tools", "done": "finalize"},
    )
    workflow.add_edge("tools", "executor_agent")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Response agent (prompt-based, same approach as single_agent.py)
# ---------------------------------------------------------------------------


def response_agent(
    state: PlannerState,
    llm: ChatOpenAI,
    formatter_prompt: str,
    max_retries: int = 1,
):
    """Format the final answer using a prompt (no ``with_structured_output``).

    Mirrors the ``ResponseAgent`` from ``single_agent.py``: invokes the
    LLM with a formatter prompt and manually parses the response into a
    ``ResponseFormatter`` with retry logic on parse failure.
    """
    messages = [
        {"role": "system", "content": formatter_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    raw_response = llm.invoke(messages).content
    formatter, parse_error = parse_response_formatter(raw_response)

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
            {"role": "assistant", "content": raw_response},
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


# ---------------------------------------------------------------------------
# Main graph constructor
# ---------------------------------------------------------------------------


def construct_multi_agent_graph(
    llm: ChatOpenAI,
    planner_prompt: str = default_planner_prompt,
    executor_prompt: str = default_executor_prompt,
    executor_tools: list = None,
    structured_output: bool = False,
    formatter_prompt: str = default_formatter_prompt,
    max_retries: int = 1,
):
    """Construct the planner-executor graph using the Send() pattern.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model shared by all agents.
    planner_prompt : str
        System prompt for the planner agent.
    executor_prompt : str
        System prompt for each executor subgraph.
    executor_tools : list
        Tools available to executor agents (LangChain tools or MCP tools).
    structured_output : bool
        If ``True``, route to ``ResponseAgent`` for structured formatting
        before ending.  If ``False``, the workflow ends directly after the
        planner decides ``FINISH``.
    formatter_prompt : str
        System prompt for the ``ResponseAgent`` (used only when
        ``structured_output=True``).
    max_retries : int
        Number of LLM retry attempts when the planner or response agent
        fails to parse its output, by default 1.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph state graph.
    """
    if executor_tools is None:
        from chemgraph.tools.ase_tools import run_ase, extract_output_json
        from chemgraph.tools.cheminformatics_tools import (
            molecule_name_to_smiles,
            smiles_to_coordinate_file,
        )
        from chemgraph.tools.generic_tools import calculator

        executor_tools = [
            molecule_name_to_smiles,
            smiles_to_coordinate_file,
            run_ase,
            extract_output_json,
            calculator,
        ]

    checkpointer = MemorySaver()

    # Build the executor subgraph
    executor_subgraph = construct_executor_subgraph(
        llm, executor_tools, executor_prompt
    )

    # Build the main graph
    graph_builder = StateGraph(PlannerState)

    # -- Nodes --
    graph_builder.add_node(
        "Planner",
        lambda state: planner_agent(
            state, llm, planner_prompt, max_retries=max_retries
        ),
    )
    graph_builder.add_node("executor_subgraph", executor_subgraph)
    graph_builder.add_node("human_review", human_review_node)

    # Conditional destinations list for the planner router
    conditional_targets = ["executor_subgraph", "human_review", END]

    if structured_output:
        graph_builder.add_node(
            "ResponseAgent",
            lambda state: response_agent(
                state,
                llm,
                formatter_prompt=formatter_prompt,
                max_retries=max_retries,
            ),
        )
        conditional_targets.append("ResponseAgent")

    # -- Edges --
    graph_builder.set_entry_point("Planner")

    graph_builder.add_conditional_edges(
        "Planner",
        partial(unified_planner_router, structured_output=structured_output),
        conditional_targets,
    )

    # Executors feed results back to the planner
    graph_builder.add_edge("executor_subgraph", "Planner")

    # After human clarification, return to the planner for re-planning
    graph_builder.add_edge("human_review", "Planner")

    if structured_output:
        graph_builder.add_edge("ResponseAgent", END)

    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Multi-agent graph (Send pattern) constructed successfully")
    return graph
