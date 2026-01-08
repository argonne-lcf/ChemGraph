from typing import TypedDict, Annotated, Any, Literal

from pydantic import BaseModel, Field
from langgraph.graph import add_messages


def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer to merge dictionaries (for worker logs)."""
    return {**a, **b}


class ExecutorState(TypedDict):
    messages: Annotated[list, add_messages]
    executor_id: str
    task_prompt: str
    inner_messages: Annotated[list, add_messages]
    next_worker_instruction: str


class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    next_step: Literal[
        "batch_orchestrator",
        "executor_subgraph",
        "insight_analyst",
        "FINISH",
    ]
    tasks: list[dict[str, Any]]
    executor_results: Annotated[list, add_messages]
    executor_logs: Annotated[dict[str, list], merge_dicts]


class ExecutorTask(BaseModel):
    """
    Represents a task assigned to an executor agent for performing tool-based computations.

    Attributes:
        task_index (int): The index or ID of the task, typically used to track execution order.
        prompt (str): A natural language prompt that describes the task or request for which
                      the executor is expected to generate tool calls.
    """

    task_index: int = Field(
        description="Task index",
    )
    prompt: str = Field(
        description="Prompt to send to executor for tool calls",
    )


class PlannerResponse(BaseModel):
    """
    Response model from the Task Decomposer agent containing a list of tasks.

    Attributes:
        tasks (list[WorkerTask]): A list of tasks that are to be assigned
        to executor agents for tool execution or computation.
    """

    thought_process: str = Field(
        description="Your reasoning for the current decision. If delegating to an agent, provide specific instructions here."
    )
    next_step: Literal[
        "batch_orchestrator",
        "executor_subgraph",
        "insight_analyst",
    ] = Field(description="The next node to activate in the workflow.")
    tasks: list[ExecutorTask] = Field(
        description="List of task to assign for executor",
        default=None,
    )


class SubPlannerDecision(BaseModel):
    """Output schema for the Sub-Planner's decision."""

    next_step: Literal["delegate_to_executor", "finish"] = Field(
        description="Check if more info is needed (delegate) or if the task is done (finish)."
    )
    instruction: str = Field(
        description="If delegating, the precise instruction for the Executor. If finishing, the final answer."
    )
