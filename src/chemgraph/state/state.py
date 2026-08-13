from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langgraph.managed.is_last_step import RemainingSteps


class State(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps