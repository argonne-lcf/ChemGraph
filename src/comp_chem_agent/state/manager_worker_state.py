from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langgraph.managed.is_last_step import RemainingSteps


class ManagerWorkerState(TypedDict):
    messages: Annotated[list, add_messages]
    worker_state: Annotated[list, add_messages]
    worker_result: Annotated[list, add_messages]
    current_task_index: int
    task_list: list
    worker_channel: dict[str, Annotated[list[str], add_messages]]
    current_worker: str
