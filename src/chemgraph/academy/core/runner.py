"""Pluggable agentic-loop runner protocol.

Academy needs to drive an agentic reasoning step (LangGraph today,
potentially ReAct/CrewAI/etc. tomorrow) without taking a hard import
on any specific implementation. The runner is injected at the campaign
boundary so the academy package itself has zero coupling to chemgraph
or any other reasoning-loop library.

Contract:

    runner = TurnRunner(
        query=...,
        tools=[...],
        model_name=..., base_url=..., api_key=..., argo_user=...,
        system_prompt=..., recursion_limit=...,
        thread_id=..., terminal_tool_names=(...),
        on_event=callback,
    ) -> awaitable[TurnRunResult]

The default runner is built in
``swarm.adapters.langgraph_runner`` (lives outside this
package's import tree to keep the boundary clean) and is the one
ChemGraphAgent uses today. Future runtimes pass their own callable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Any, Protocol


class TurnRunResult(Protocol):
    """Just the fields academy actually reads from a turn result."""

    final_text: str
    executed_tool_names: tuple[str, ...]
    terminal_tool: str | None
    thread_id: str


TurnRunner = Callable[..., Awaitable[TurnRunResult]]
"""Async callable that drives one reasoning-loop turn.

Implementations accept the kwargs documented in this module's docstring
and return any object satisfying ``TurnRunResult``. Implementations are
free to add extra kwargs; the academy caller passes only the ones below.
"""


__all__ = ["TurnRunner", "TurnRunResult"]
