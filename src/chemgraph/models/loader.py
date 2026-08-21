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
from chemgraph.models.endpoints import alcf as alcf_ep
from chemgraph.models.endpoints import anthropic_direct as anthropic_ep
from chemgraph.models.endpoints import argo as argo_ep
from chemgraph.models.endpoints import codex as codex_ep
from chemgraph.models.endpoints import google_direct as google_ep
from chemgraph.models.endpoints import groq as groq_ep
from chemgraph.models.endpoints import ollama as ollama_ep
from chemgraph.models.endpoints import openai_direct as openai_ep
from chemgraph.models.endpoints import openrouter as openrouter_ep
from chemgraph.models.endpoints import vllm as vllm_ep
from chemgraph.models.settings import LLMSettings

# Ordered endpoint registry. Resolution preserves the historical dispatch order:
# codex -> openrouter -> Argo Anthropic -> Argo OpenAI -> curated ALCF ->
# direct OpenAI -> direct Anthropic -> direct Gemini -> Ollama -> groq. Prefix
# routes (``matches`` on a prefix) run before catalog checks, so names such as
# ``openrouter:openai/o3`` cannot be misclassified. The vLLM/custom fallback is
# *not* in this list; it is selected only via ``vllm.can_handle`` as a last
# resort.
_ENDPOINT_REGISTRY = (
    codex_ep.SPEC,
    openrouter_ep.SPEC,
    argo_ep.ANTHROPIC_SPEC,
    argo_ep.OPENAI_SPEC,
    alcf_ep.SPEC,
    openai_ep.SPEC,
    anthropic_ep.SPEC,
    google_ep.SPEC,
    ollama_ep.SPEC,
    groq_ep.SPEC,
)


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
        settings_reasoning = getattr(settings, "reasoning_effort", None)
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
    """Return the first endpoint spec whose ``matches`` accepts the model.

    Falls back to the vLLM/custom endpoint only when a custom base URL is
    available. Raises ``ValueError`` for an otherwise unknown model.
    """
    for spec in _ENDPOINT_REGISTRY:
        if spec.matches(request.model):
            return spec

    if vllm_ep.can_handle(request):
        return vllm_ep.SPEC

    raise ValueError(
        f"Model '{request.model}' not found in any supported model list. "
        "Use a model from: OpenAI, Anthropic, Gemini, groq:<model>, "
        "openrouter:<model>, codex:<model>, argo:<model>, ALCF, or Ollama."
    )


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
    prepared = spec.prepare(request)
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
