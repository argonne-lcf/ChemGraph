"""The only production construction site for ``ChatGoogleGenerativeAI``."""

from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


def build(client_kwargs: dict[str, Any]) -> ChatGoogleGenerativeAI:
    """Construct a ``ChatGoogleGenerativeAI`` from prepared keyword arguments."""
    return ChatGoogleGenerativeAI(**client_kwargs)
