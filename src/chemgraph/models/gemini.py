"""Deprecated compatibility shim for Gemini model loading.

Construction now lives in ``chemgraph.models.endpoints.google_direct`` behind
the Google-native protocol builder. Prefer
``chemgraph.models.loader.load_chat_model``.
"""

from __future__ import annotations

import warnings

from langchain_google_genai import ChatGoogleGenerativeAI

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints import google_direct as _google_direct
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_DEPRECATION = (
    "load_gemini_model is deprecated; use "
    "chemgraph.models.loader.load_chat_model instead."
)


def load_gemini_model(
    model_name: str,
    temperature: float,
    api_key: str = None,
    prompt: str = None,
    base_url: str = None,
) -> ChatGoogleGenerativeAI:
    """Load a Gemini chat model (deprecated). Delegates to the endpoint."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    request = ModelRequest(
        model=model_name, temperature=temperature, api_key=api_key, base_url=base_url
    )
    return _google_direct.SPEC.build(request)
