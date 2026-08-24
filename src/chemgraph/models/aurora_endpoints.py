"""Deprecated compatibility shim for Aurora on-node model loading.

Construction now lives in ``chemgraph.models.endpoints.aurora`` behind the
shared OpenAI-compatible protocol builder. Prefer
``chemgraph.models.loader.load_chat_model``.

``AURORA_MODEL_PREFIX`` and ``_normalize_aurora_model`` are re-exported for one
release so existing callers and tests keep working.
"""

from __future__ import annotations

import warnings

from langchain_openai import ChatOpenAI

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints import aurora as _aurora
from chemgraph.utils.logging_config import setup_logger

# Re-exported for backward compatibility.
from chemgraph.models.endpoints.aurora import AURORA_PREFIX as AURORA_MODEL_PREFIX  # noqa: F401

logger = setup_logger(__name__)

_DEPRECATION = (
    "load_aurora_model is deprecated; use "
    "chemgraph.models.loader.load_chat_model instead."
)


def _normalize_aurora_model(model_name: str) -> str:
    """Strip the ``aurora:`` prefix to get the name the endpoint expects.

    Kept for backward compatibility. New code should let the endpoint spec
    handle prefix stripping via ``load_chat_model``.
    """
    if not model_name.startswith(AURORA_MODEL_PREFIX):
        return model_name
    return model_name.removeprefix(AURORA_MODEL_PREFIX)


def load_aurora_model(
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Load an Aurora on-node chat model (deprecated). Delegates to the endpoint."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    request = ModelRequest(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    return _aurora.SPEC.build(request)
