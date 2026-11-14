from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class WorkerTask(BaseModel):
    """
    Represents a task assigned to an executor agent for performing tool-based computations.

    Attributes:
        task_index (int): The index or ID of the task, typically used to track execution order.
        prompt (str): A natural language prompt that describes the task or request for which
                       the executor is expected to generate tool calls.
        retry_count (int): How many times this task has been previously attempted.
                           Defaults to 0 for new tasks.  When the planner re-dispatches
                           a failed task, the router increments this value automatically.
    """

    task_index: int = Field(..., description="Task index")
    prompt: str = Field(..., description="Prompt to send to executor for tool calls")
    retry_count: int = Field(
        default=0,
        description="Number of previous attempts for this task (0 = first attempt)",
    )


class PlannerResponse(BaseModel):
    """
    Response model from the Planner agent.

    The planner acts as a router: it decides whether to dispatch tasks
    to executor subgraphs (``executor_subgraph``), ask the human user
    for clarification (``ask_human``), or finish (``FINISH``) when all
    work is done.

    Attributes:
        thought_process (str): The planner's reasoning for the current decision.
        next_step (str): The next node to activate — ``"executor_subgraph"``
            to fan-out tasks, ``"ask_human"`` to request human input, or
            ``"FINISH"`` to end the workflow.
        tasks (list[WorkerTask] | None): Tasks to assign when routing to executors.
        clarification (str | None): Question to ask the human when
            ``next_step`` is ``"ask_human"``.
    """

    thought_process: str = Field(
        description="Your reasoning for the current decision."
    )
    next_step: Literal["executor_subgraph", "ask_human", "FINISH"] = Field(
        description="The next node to activate in the workflow."
    )
    tasks: list[WorkerTask] = Field(
        default=None,
        description="List of tasks to assign to executor subgraphs.",
    )
    clarification: Optional[str] = Field(
        default=None,
        description=(
            "Question to present to the human user when next_step is 'ask_human'. "
            "Must be provided when next_step is 'ask_human'."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_planner_payload(cls, data: Any) -> Any:
        """Accept common planner variants and coerce into PlannerResponse shape.

        Parameters
        ----------
        data : Any
            Raw planner payload before Pydantic validation.

        Returns
        -------
        Any
            Normalized payload compatible with ``PlannerResponse``.
        """
        if isinstance(data, list):
            return {
                "thought_process": "Delegating parsed tasks to executors.",
                "next_step": "executor_subgraph",
                "tasks": data,
            }

        if isinstance(data, dict):
            normalized = dict(data)
            # Accept legacy "worker_tasks" key
            if "tasks" not in normalized and "worker_tasks" in normalized:
                normalized["tasks"] = normalized.pop("worker_tasks")
            if "tasks" in normalized and "next_step" not in normalized:
                normalized["next_step"] = "executor_subgraph"
            if "tasks" in normalized and "thought_process" not in normalized:
                normalized["thought_process"] = (
                    "Delegating parsed tasks to executors."
                )
            # Accept legacy "question" key as clarification
            if (
                normalized.get("next_step") == "ask_human"
                and "clarification" not in normalized
                and "question" in normalized
            ):
                normalized["clarification"] = normalized.pop("question")
            return normalized

        return data
