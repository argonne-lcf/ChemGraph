"""Deprecated compatibility shim for OpenRouter model loading.

Construction now lives in ``chemgraph.models.endpoints.openrouter`` behind the
shared OpenAI-compatible protocol builder. Prefer
``chemgraph.models.loader.load_chat_model``.

``OPENROUTER_DEFAULT_MAX_TOKENS`` is re-exported for one release.
"""

from __future__ import annotations

import warnings

from langchain_openai import ChatOpenAI

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints import openrouter as _openrouter

# Re-exported for backward compatibility.
from chemgraph.models.endpoints.openrouter import (  # noqa: F401
    OPENROUTER_DEFAULT_MAX_TOKENS,
)
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_DEPRECATION = (
    "load_openrouter_model is deprecated; use "
    "chemgraph.models.loader.load_chat_model instead."
)


def load_openrouter_model(
    model_name: str,
    temperature: float = 0.0,
    api_key: str = None,
    base_url: str = None,
    max_tokens: int = OPENROUTER_DEFAULT_MAX_TOKENS,
) -> ChatOpenAI:
    """Load an OpenRouter chat model (deprecated). Delegates to the endpoint."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    request = ModelRequest(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    return _openrouter.SPEC.build(request)
