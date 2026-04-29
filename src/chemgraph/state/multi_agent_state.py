import operator
from typing import TypedDict, Annotated, Any, Literal

from langgraph.graph import add_messages


def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer that merges dictionaries (used for executor logs)."""
    return {**a, **b}


class ExecutorState(TypedDict):
    """Local state for each executor subgraph spawned via Send().

    Each executor instance gets its own isolated copy of this state.
    The ``messages`` list holds the executor's ReAct conversation
    (system prompt, LLM responses, tool calls/results).

    ``task_index`` identifies which planner task this executor is
    working on, enabling the planner to correlate results with tasks.

    ``retry_count`` tracks how many times this particular task has
    been retried, so the planner and router can enforce retry limits.
    """

    messages: Annotated[list, add_messages]
    executor_id: str
    task_index: int
    retry_count: int


class PlannerState(TypedDict):
    """Global state for the main planner-executor graph.

    The planner reads ``messages`` (the original user query plus its own
    prior outputs) and ``executor_results`` (merged results from all
    completed executor subgraphs) to decide what to do next.

    ``planner_iterations`` tracks how many times the planner has
    dispatched tasks to executors, providing a guard against infinite
    Planner -> Executor -> Planner cycles.

    ``failed_tasks`` accumulates structured records of executor
    failures across iterations, enabling the planner to make informed
    retry decisions.  Each entry is a dict with keys: ``task_index``,
    ``error``, ``retry_count``, and ``executor_id``.
    """

    messages: Annotated[list, add_messages]
    next_step: Literal["executor_subgraph", "FINISH"]
    tasks: list[dict[str, Any]]
    executor_results: Annotated[list, operator.add]
    executor_logs: Annotated[dict[str, list], merge_dicts]
    planner_iterations: int
    failed_tasks: Annotated[list, operator.add]
