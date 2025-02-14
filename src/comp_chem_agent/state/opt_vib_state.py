from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class MultiAgentState(TypedDict):
    question: str
    first_router_response: Annotated[list, add_messages]
    regular_response: Annotated[list, add_messages]
    feedback_response: Annotated[list, add_messages]
    geometry_response: Annotated[list, add_messages]
    parameter_response: Annotated[list, add_messages]
    opt_response: Annotated[list, add_messages]
    end_response: Annotated[list, add_messages]
