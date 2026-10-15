"""Explicit approval modes shared by the workspace workflows."""

from typing import Literal


ApprovalMode = Literal["review", "bypass", "scoped"]


def normalize_approval_mode(workflow, mode=None, *, deepagent_auto_approve=False) -> ApprovalMode:
    """Resolve the legacy standalone option without silently changing policies."""
    if mode not in (None, "review", "bypass", "scoped"):
        raise ValueError("approval_mode must be 'review', 'bypass' or 'scoped'.")
    if deepagent_auto_approve:
        if workflow != "deep_agent":
            raise ValueError("deepagent_auto_approve is supported only for the deep_agent workflow.")
        if mode in {"review", "scoped"}:
            raise ValueError(f"deepagent_auto_approve conflicts with approval_mode={mode!r}.")
        mode = "bypass"
    if mode == "bypass" and workflow not in {"main_agent", "deep_agent"}:
        raise ValueError("Approval bypass requires the main_agent or deep_agent workflow.")
    # Scoped runs review through chemgraph.runtime; see ScopedToolsMiddleware.
    if mode == "scoped" and workflow != "main_agent":
        raise ValueError("Scoped approval requires the main_agent workflow.")
    return mode or "review"
