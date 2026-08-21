"""Deprecated compatibility shim for ALCF model loading.

Construction now lives in ``chemgraph.models.endpoints.alcf`` behind the shared
OpenAI-compatible protocol builder. Prefer
``chemgraph.models.loader.load_chat_model``.

The ``_normalize_alcf_model`` helper is re-exported for one release.
"""

from __future__ import annotations

import warnings

from langchain_openai import ChatOpenAI

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints import alcf as _alcf

# Re-exported for backward compatibility.
from chemgraph.models.endpoints.alcf import (  # noqa: F401
    ALCF_MODEL_PREFIX,
    _normalize_alcf_model,
)
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_DEPRECATION = (
    "load_alcf_model is deprecated; use "
    "chemgraph.models.loader.load_chat_model instead."
)


def load_alcf_model(
    model_name: str,
    base_url: str = None,
    api_key: str = None,
) -> ChatOpenAI:
    """Load an ALCF model (deprecated). Delegates to the endpoint."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    request = ModelRequest(model=model_name, base_url=base_url, api_key=api_key)
    return _alcf.SPEC.build(request)
