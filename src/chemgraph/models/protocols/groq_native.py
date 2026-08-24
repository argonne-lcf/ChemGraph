"""The only production construction site for the Groq client.

Groq keeps its own loader (``load_groq_model``) with its API-key handling and
retry-on-auth logic. This builder is the sole call site: the endpoint packs the
loader's arguments into ``client_kwargs`` and hands them here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from chemgraph.models.groq import load_groq_model


def build(client_kwargs: dict[str, Any]) -> BaseChatModel:
    """Construct a Groq chat model from prepared keyword arguments."""
    return load_groq_model(**client_kwargs)
