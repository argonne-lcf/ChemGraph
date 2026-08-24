"""The only production construction site for the Codex client.

Codex keeps its own loader (``load_codex_model``) with its subscription
authentication. This builder is the sole call site: the endpoint packs the
loader's arguments into ``client_kwargs`` and hands them here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from chemgraph.models.codex import load_codex_model


def build(client_kwargs: dict[str, Any]) -> BaseChatModel:
    """Construct a Codex chat model from prepared keyword arguments."""
    return load_codex_model(**client_kwargs)
