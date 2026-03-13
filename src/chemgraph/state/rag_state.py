"""LangGraph state definition for the RAG agent workflow."""

from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages
from langgraph.managed.is_last_step import RemainingSteps


class RAGState(TypedDict):
    """State for the RAG agent workflow.

    Extends the base message-passing state with fields to track
    the loaded document path and retrieved context.

    Attributes
    ----------
    messages : list
        Accumulated conversation messages (managed by LangGraph).
    remaining_steps : RemainingSteps
        Counter for recursion-limit enforcement.
    document_path : str or None
        Path to the currently loaded document, if any.
    retrieved_context : str or None
        The most recently retrieved context from the vector store,
        injected into the agent's prompt for grounded answers.
    """

    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps
    document_path: Optional[str]
    retrieved_context: Optional[str]
