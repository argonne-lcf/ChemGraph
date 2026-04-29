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
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

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
    failed_tasks = state.get("failed_tasks", [])
    content_block = f"Current Conversation History: {state['messages']}"
    if executor_outputs:
        results_text = "\n".join(
            m.content if hasattr(m, "content") else str(m) for m in executor_outputs
        )
        content_block += (
            f"\n\n### UPDATED: Results from Executor Tasks ###\n{results_text}"
        )
    if failed_tasks:
        failure_lines = []
        for ft in failed_tasks:
            failure_lines.append(
                f"- Task {ft.get('task_index', '?')} "
                f"(retry #{ft.get('retry_count', 0)}): "
                f"{ft.get('error', 'unknown error')}"
            )
        content_block += (
            "\n\n### FAILED TASKS (may be retried) ###\n"
            + "\n".join(failure_lines)
            + "\n\nYou may retry failed tasks by including them in your "
            "tasks list with the same task_index. Use the error information "
            "above to adjust the prompt if needed (e.g., fix molecule names, "
            "adjust parameters). If a task cannot succeed, set next_step "
            "to FINISH and explain the failure in thought_process."
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
    return {
        "messages": [AIMessage(content=response_obj.thought_process)],
        "next_step": response_obj.next_step,
        "tasks": response_obj.tasks if response_obj.tasks else [],
        "planner_iterations": current_iterations + 1,
    }


# ---------------------------------------------------------------------------
# Planner router (conditional edge)
# ---------------------------------------------------------------------------


def unified_planner_router(
    state: PlannerState,
    structured_output: bool = False,
    max_planner_iterations: int = 3,
    max_task_retries: int = 2,
) -> Union[str, list[Send]]:
    """Route based on the planner's ``next_step`` decision.

    * ``executor_subgraph`` -- fan-out tasks via ``Send()``
    * ``FINISH`` -- go to ``ResponseAgent`` (if structured_output) or ``END``

    A cycle guard forces ``FINISH`` when the planner has dispatched
    executors ``max_planner_iterations`` times to prevent infinite loops.

    For retried tasks, the ``retry_count`` from the ``WorkerTask`` is
    checked against ``max_task_retries``.  Tasks that have exceeded the
    retry limit are skipped and logged as permanently failed.
    """
    next_step = state.get("next_step")
    iterations = state.get("planner_iterations", 0)

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

        # Build a lookup of previous failure counts from state.
        # This covers cases where the planner emits a task without
        # explicitly setting retry_count — we infer it from history.
        failed_history: dict[int, int] = {}
        for ft in state.get("failed_tasks", []):
            tidx = ft.get("task_index", -1)
            prev = ft.get("retry_count", 0)
            # Track the highest retry_count seen for each task_index
            failed_history[tidx] = max(failed_history.get(tidx, 0), prev + 1)

        sends = []
        for i, t in enumerate(tasks):
            task_index = getattr(t, "task_index", i + 1)
            # Determine retry_count: use whichever is larger —
            # the value from the task object or the inferred history.
            task_retry = getattr(t, "retry_count", 0)
            inferred_retry = failed_history.get(task_index, 0)
            retry_count = max(task_retry, inferred_retry)

            if retry_count >= max_task_retries:
                logger.warning(
                    "Task %d exceeded max retries (%d); skipping.",
                    task_index,
                    max_task_retries,
                )
                continue

            sends.append(
                Send(
                    "executor_subgraph",
                    {
                        "executor_id": f"worker_{task_index}",
                        "task_index": task_index,
                        "retry_count": retry_count,
                        "messages": [getattr(t, "prompt", str(t))],
                    },
                )
            )

        if not sends:
            # All tasks were skipped (max retries exceeded).
            logger.warning(
                "All dispatched tasks exceeded retry limits; forcing FINISH."
            )
            if structured_output:
                return "ResponseAgent"
            return END

        return sends

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


_ERROR_MARKERS = [
    "Error:",
    "error:",
    "Exception:",
    "exception:",
    "Traceback",
    "failed",
    "FAILED",
    "could not",
    "Could not",
    "No PubChem compound found",
    "ValueError",
    "TypeError",
    "KeyError",
    "RuntimeError",
]


def _detect_executor_failure(messages: list) -> tuple[bool, str | None]:
    """Scan executor message history for signs of failure.

    Checks for:
    1. ``ToolMessage`` objects with ``status == "error"``
       (produced by ``ToolNode(handle_tool_errors=True)``).
    2. Error markers in the final assistant message content.

    Returns ``(is_failed, error_summary)``.
    """
    # Collect all tool-level errors
    tool_errors = []
    for m in messages:
        if isinstance(m, ToolMessage):
            if getattr(m, "status", None) == "error":
                tool_errors.append(m.content)

    if tool_errors:
        return True, "; ".join(tool_errors)

    # Check the final message for error markers
    final = messages[-1] if messages else None
    if final is not None:
        content = getattr(final, "content", str(final))
        if isinstance(content, str):
            for marker in _ERROR_MARKERS:
                if marker in content:
                    # Only flag as failure if the executor itself reports failure,
                    # not if it's merely describing a prior error it recovered from.
                    # Heuristic: if the last message also contains "success" or
                    # "result", treat it as a recovered scenario.
                    lower = content.lower()
                    if "success" not in lower and "result:" not in lower:
                        return True, content[:500]

    return False, None


def format_executor_output(state: ExecutorState) -> dict:
    """Bridge: convert local ``ExecutorState`` into a ``PlannerState`` update.

    Writes the executor's final answer into ``executor_results`` and
    its full message history into ``executor_logs`` so the planner can
    inspect them on the next iteration.

    Detects executor failures by scanning the message history for tool
    errors and error markers.  When a failure is detected, populates
    ``failed_tasks`` so the planner can decide whether to retry.
    """
    executor_id = state["executor_id"]
    task_index = state.get("task_index", -1)
    retry_count = state.get("retry_count", 0)
    final_message = state["messages"][-1].content
    full_history = state["messages"]

    is_failed, error_summary = _detect_executor_failure(list(state["messages"]))

    result: dict = {
        "executor_logs": {executor_id: full_history},
    }

    if is_failed:
        logger.warning(
            "Executor %s (task_index=%d, retry=%d) FAILED: %s",
            executor_id,
            task_index,
            retry_count,
            error_summary,
        )
        result["executor_results"] = [
            f"[{executor_id}] FAILED (task_index={task_index}, "
            f"retry={retry_count}): {error_summary}"
        ]
        result["failed_tasks"] = [
            {
                "task_index": task_index,
                "executor_id": executor_id,
                "error": error_summary,
                "retry_count": retry_count,
            }
        ]
    else:
        result["executor_results"] = [
            f"[{executor_id}] Result (task_index={task_index}): {final_message}"
        ]
        result["failed_tasks"] = []

    return result


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
    workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
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
    max_task_retries: int = 2,
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
    max_task_retries : int
        Maximum number of times a single executor task may be retried
        after failure.  Once a task reaches this limit, the router skips
        it and the planner must finish without it, by default 2.

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

    # Conditional destinations list for the planner router
    conditional_targets = ["executor_subgraph", END]

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
        partial(
            unified_planner_router,
            structured_output=structured_output,
            max_task_retries=max_task_retries,
        ),
        conditional_targets,
    )

    # Executors feed results back to the planner
    graph_builder.add_edge("executor_subgraph", "Planner")

    if structured_output:
        graph_builder.add_edge("ResponseAgent", END)

    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Multi-agent graph (Send pattern) constructed successfully")
    return graph
