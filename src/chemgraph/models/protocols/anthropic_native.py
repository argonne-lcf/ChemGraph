"""The only production construction site for ``ChatAnthropic``."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic


def build(client_kwargs: dict[str, Any]) -> ChatAnthropic:
    """Construct a ``ChatAnthropic`` from prepared keyword arguments."""
    return ChatAnthropic(**client_kwargs)
