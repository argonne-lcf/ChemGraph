"""Shared model-loading utility for ChemGraph.

Provides a single ``load_chat_model`` function that detects the endpoint for a
given model name and returns a LangChain ``BaseChatModel`` instance. This is the
only model-routing entry point; the pipeline is::

    caller -> load_chat_model -> endpoint spec -> protocol builder -> client

Endpoint specs own base URLs, credentials, model-name transforms, defaults, and
per-model quirks. Protocol builders own the sole client-construction sites.
"""

from __future__ import annotations

from typing import Optional

from chemgraph.models.endpoints import ModelRequest, PreparedModel
from chemgraph.models.endpoints.registry import select_endpoint
from chemgraph.models.settings import LLMSettings


def _build_request(
    model_name: str | None,
    temperature: float,
    base_url: Optional[str],
    api_key: Optional[str],
    argo_user: Optional[str],
    reasoning_effort: Optional[str],
    settings: LLMSettings | None,
) -> ModelRequest:
    """Resolve explicit arguments and settings into a ``ModelRequest``.

    Settings take precedence over the individual arguments when supplied,
    preserving the historical behavior of ``load_chat_model``.
    """
    if settings is not None:
        model_name = settings.model
        base_url = settings.base_url
        api_key = settings.api_key
        argo_user = settings.argo_user
        if settings.temperature is not None:
            temperature = settings.temperature
        settings_reasoning = settings.reasoning_effort
        if settings_reasoning is not None:
            reasoning_effort = settings_reasoning

    if model_name is None:
        raise ValueError("load_chat_model requires model_name or settings")

    return ModelRequest(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        argo_user=argo_user,
        reasoning_effort=reasoning_effort,
        settings=settings,
    )


def _select_endpoint(request: ModelRequest):
    """Backward-compatible alias for the shared endpoint selector."""
    return select_endpoint(request)


def load_chat_model_prepared(
    model_name: str | None = None,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    argo_user: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    *,
    settings: LLMSettings | None = None,
) -> tuple["object", PreparedModel]:
    """Load a chat model and return it alongside its resolved metadata.

    Returns a ``(client, PreparedModel)`` tuple so callers such as
    ``ChemGraph`` can read the effective reasoning effort and structured-output
    capability without re-deriving provider facts. ``load_chat_model`` is the
    thin wrapper that returns only the client.
    """
    request = _build_request(
        model_name,
        temperature,
        base_url,
        api_key,
        argo_user,
        reasoning_effort,
        settings,
    )
    spec = _select_endpoint(request)
    prepared = spec.prepare_request(request)
    client = spec.protocol_build(prepared.client_kwargs)
    return client, prepared


def load_chat_model(
    model_name: str | None = None,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    argo_user: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    *,
    settings: LLMSettings | None = None,
):
    """Load a LangChain chat model by endpoint auto-detection.

    Parameters
    ----------
    model_name : str, optional
        Model name from any supported endpoint.
    temperature : float
        Sampling temperature (default 0.0 for deterministic output).
    base_url : str, optional
        Endpoint base URL override.
    api_key : str, optional
        API key override (falls back to endpoint-specific environment variables).
    argo_user : str, optional
        Argo user identifier.
    reasoning_effort : str, optional
        Reasoning effort for models that support it (validated per endpoint).
    settings : LLMSettings, optional
        Canonical endpoint settings. When provided, this overrides
        model_name/base_url/api_key/argo_user/reasoning_effort.

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance.

    Raises
    ------
    ValueError
        If the model name is not found in any supported endpoint.
    """
    client, _ = load_chat_model_prepared(
        model_name=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        argo_user=argo_user,
        reasoning_effort=reasoning_effort,
        settings=settings,
    )
    return client
