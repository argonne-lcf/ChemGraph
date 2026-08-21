"""Deprecated compatibility shim for OpenAI / Argo model loading.

The construction logic now lives in ``chemgraph.models.endpoints.openai_direct``
and ``chemgraph.models.endpoints.argo`` behind shared protocol builders. Prefer
``chemgraph.models.loader.load_chat_model``, which routes Argo Claude models
through the Anthropic-native protocol.

This module keeps the previous public surface -- ``load_openai_model`` and the
``_normalize_argo_model`` helper -- for one release.
"""

from __future__ import annotations

import warnings

from langchain_openai import ChatOpenAI

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints import argo as _argo
from chemgraph.models.endpoints import openai_direct as _openai_direct

# Re-exported for backward compatibility (tests and external callers import it).
from chemgraph.models.endpoints.argo import (  # noqa: F401
    ARGO_LOCAL_OPENAI_MODEL_MAP,
    ARGO_MODEL_MAP,
    _normalize_argo_model,
)
from chemgraph.models.endpoints.base import is_local_http_endpoint as _is_local_http_endpoint  # noqa: F401
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_DEPRECATION = (
    "load_openai_model is deprecated; use "
    "chemgraph.models.loader.load_chat_model instead."
)


def load_openai_model(
    model_name: str,
    temperature: float,
    api_key: str = None,
    prompt: str = None,
    base_url: str = None,
    argo_user: str = None,
    reasoning_effort: str = None,
) -> ChatOpenAI:
    """Load an OpenAI or Argo chat model (deprecated).

    Delegates to the ``openai_direct`` or Argo OpenAI-compatible endpoint. The
    historical ChatOpenAI return type and signature are preserved for backward
    compatibility.
    """
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)

    request = ModelRequest(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        argo_user=argo_user,
        reasoning_effort=reasoning_effort,
    )
    endpoint = _argo.SPEC if model_name.startswith("argo:") else _openai_direct.SPEC
    return endpoint.build(request)
