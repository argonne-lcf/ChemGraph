from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
