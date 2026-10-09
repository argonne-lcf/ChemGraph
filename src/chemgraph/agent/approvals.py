"""Explicit approval modes shared by the workspace workflows."""

from typing import Literal


ApprovalMode = Literal["review", "bypass"]


def normalize_approval_mode(workflow, mode=None, *, deepagent_auto_approve=False) -> ApprovalMode:
    """Resolve the legacy standalone option without silently changing policies."""
    if mode not in (None, "review", "bypass"):
        raise ValueError("approval_mode must be 'review' or 'bypass'.")
    if deepagent_auto_approve:
        if workflow != "deep_agent":
            raise ValueError("deepagent_auto_approve is supported only for the deep_agent workflow.")
        if mode == "review":
            raise ValueError("deepagent_auto_approve conflicts with approval_mode='review'.")
        mode = "bypass"
    if mode == "bypass" and workflow not in {"main_agent", "deep_agent"}:
        raise ValueError("Approval bypass requires the main_agent or deep_agent workflow.")
    return mode or "review"
