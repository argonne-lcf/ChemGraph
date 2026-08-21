"""The only production construction site for ``ChatOpenAI``.

Every OpenAI-compatible endpoint (OpenAI direct, both Argo variants, all three
ALCF clusters, OpenRouter, and vLLM/custom endpoints) reaches this builder. It
receives fully prepared kwargs and applies no endpoint-specific conditions.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI


def build(client_kwargs: dict[str, Any]) -> ChatOpenAI:
    """Construct a ``ChatOpenAI`` from prepared keyword arguments."""
    return ChatOpenAI(**client_kwargs)
