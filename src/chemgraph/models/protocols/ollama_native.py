"""The only production construction site for the Ollama client.

Ollama keeps its own loader (``load_ollama_model``) with its curated-catalog
validation. This builder is the sole call site: the endpoint packs the loader's
arguments into ``client_kwargs`` and hands them here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from chemgraph.models.local_model import load_ollama_model


def build(client_kwargs: dict[str, Any]) -> BaseChatModel:
    """Construct an Ollama chat model from prepared keyword arguments."""
    return load_ollama_model(**client_kwargs)
