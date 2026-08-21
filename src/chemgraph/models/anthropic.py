"""Deprecated compatibility shim for Anthropic model loading.

Construction now lives in ``chemgraph.models.endpoints.anthropic_direct`` behind
the Anthropic-native protocol builder. Prefer
``chemgraph.models.loader.load_chat_model``.
"""

from __future__ import annotations

import warnings

from langchain_anthropic import ChatAnthropic

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints import anthropic_direct as _anthropic_direct
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_DEPRECATION = (
    "load_anthropic_model is deprecated; use "
    "chemgraph.models.loader.load_chat_model instead."
)


def load_anthropic_model(
    model_name: str, temperature: float, api_key: str = None, prompt: str = None
) -> ChatAnthropic:
    """Load an Anthropic chat model (deprecated). Delegates to the endpoint."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    request = ModelRequest(
        model=model_name, temperature=temperature, api_key=api_key
    )
    return _anthropic_direct.SPEC.build(request)
