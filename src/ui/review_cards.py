"""Deep Agent action-review cards for the Streamlit chat.

The CLI renders each pending tool action (shell command, file write, file
edit, ...) as a panel and asks for approve/reject.  This module gives the
chat the same information as cards with buttons.  ``action_preview`` is
pure and mirrors ``chemgraph.cli.formatting.build_action_review`` (same
sanitization, same previews, no host-file reads); ``render_action_card``
is the Streamlit wrapper.
"""

from __future__ import annotations

import json
from difflib import unified_diff
from typing import Any, Optional

import streamlit as st

from chemgraph.agent.interrupts import is_tool_review
from chemgraph.cli.formatting import _limit_review_text, _safe_review_text

#: Preview budget before the full arguments move behind an expander.
PREVIEW_MAX_LINES = 40
PREVIEW_MAX_CHARS = 8000


def allowed_decisions(payload: dict, action_name: str) -> list[str]:
    """Return the approve/reject decisions permitted for *action_name*.

    Parameters
    ----------
    payload : dict
        Tool-review interrupt payload (``action_requests`` + ``review_configs``).
    action_name : str
        Name of the action being reviewed.

    Returns
    -------
    list[str]
        Subset of ``["approve", "reject"]`` in the review config's order.
    """
    for item in payload.get("review_configs", []):
        if isinstance(item, dict) and item.get("action_name") == action_name:
            return [
                decision
                for decision in item.get("allowed_decisions", [])
                if decision in {"approve", "reject"}
            ]
    return []


def action_preview(action: dict) -> dict[str, Any]:
    """Describe one action request for display.

    Parameters
    ----------
    action : dict
        ``{"name": str, "args": dict, ...}`` from ``action_requests``.

    Returns
    -------
    dict
        ``name`` (str), ``summary`` (one-line identity: tool and path),
        ``blocks`` (list of ``(label, text, language)`` previews, already
        sanitized and bounded), ``truncated`` (bool) and ``arguments``
        (full JSON of the arguments for the expander).
    """
    name = str(action.get("name", "unknown"))
    args = action.get("args", {})
    label = ""
    preview: list[tuple[str, Optional[str]]] = []
    remaining = dict(args) if isinstance(args, dict) else {}
    if isinstance(args, dict):
        if name == "execute" and isinstance(args.get("command"), str):
            label = "Command"
            preview = [(remaining.pop("command"), "bash")]
        elif name == "write_file" and isinstance(args.get("content"), str):
            label = "Content"
            preview = [(remaining.pop("content"), None)]
        elif name == "edit_file" and all(
            isinstance(args.get(key), str) for key in ("old_string", "new_string")
        ):
            before = _safe_review_text(remaining.pop("old_string")).splitlines(keepends=True)
            after = _safe_review_text(remaining.pop("new_string")).splitlines(keepends=True)
            lines = unified_diff(
                before, after, fromfile="before", tofile="after",
                n=max(len(before), len(after)),
            )
            diff = "".join(
                line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                for line in lines
            )
            label = "Proposed replacement"
            if args.get("replace_all") is True:
                label += " (applies to all occurrences)"
            preview = [(diff or "(No changes)", "diff")]

    try:
        arguments = json.dumps(args, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        arguments = repr(args)
    arguments = _safe_review_text(arguments)

    blocks: list[tuple[str, str, Optional[str]]] = []
    truncated = False
    if remaining or not preview:
        try:
            rest = json.dumps(remaining, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rest = repr(remaining)
        text, omitted = _limit_review_text(
            _safe_review_text(rest), PREVIEW_MAX_LINES, PREVIEW_MAX_CHARS
        )
        truncated |= omitted
        blocks.append(("Arguments", text, "json"))
    for value, language in preview:
        text, omitted = _limit_review_text(
            _safe_review_text(value), PREVIEW_MAX_LINES, PREVIEW_MAX_CHARS
        )
        truncated |= omitted
        blocks.append((label, text, language))

    summary_parts = [f"Tool: {name}"]
    if isinstance(args, dict) and "file_path" in args:
        summary_parts.append(f"Path: {args['file_path']}")
    summary = _safe_review_text(" | ".join(summary_parts)).replace("\n", "\\n")
    if len(summary) > 240:
        summary = summary[:120] + " … " + summary[-120:]

    return {
        "name": name,
        "summary": summary,
        "blocks": blocks,
        "truncated": truncated,
        "arguments": arguments,
    }


def review_summary(payload: Any) -> str:
    """One-line description of a review interrupt for transcripts."""
    if not is_tool_review(payload):
        return str(payload)
    names = [
        str(action.get("name", "unknown"))
        for action in payload.get("action_requests", [])
        if isinstance(action, dict)
    ]
    count = len(names)
    noun = "action" if count == 1 else "actions"
    return f"Review {count} Deep Agent {noun}: " + ", ".join(names)


def render_action_card(action: dict, index: int, total: int, key: str) -> None:
    """Render one action request as a bordered card.

    Parameters
    ----------
    action : dict
        Action request from the interrupt payload.
    index : int
        One-based position among the pending actions.
    total : int
        Total number of pending actions.
    key : str
        Unique widget-key prefix for the card's expander.
    """
    info = action_preview(action)
    with st.container(border=True):
        st.markdown(f"**Review action {index} of {total}** — `{info['summary']}`")
        for label, text, language in info["blocks"]:
            st.caption(label)
            st.code(text, language=language or "text")
        if info["truncated"]:
            st.caption("Preview shortened.")
        with st.expander("Full arguments", expanded=False):
            st.code(info["arguments"], language="json")


__all__ = [
    "action_preview",
    "allowed_decisions",
    "render_action_card",
    "review_summary",
]
