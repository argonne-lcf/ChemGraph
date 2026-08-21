"""Argo endpoints (hosted Argo API and shim/proxy/custom OpenAI-compatible).

Owns everything Argo-specific: the wire-name maps, ``CHEMGRAPH_ARGO_MODEL_FORMAT``
handling, localhost detection, prefix removal, Claude streaming, minimal-parameter
models, Argo ``user`` payload, and the ``supports_structured_output=False``
capability. All logic is moved verbatim from the previous ``models/openai.py``.
"""

from __future__ import annotations

import os

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
    is_local_http_endpoint,
)
from chemgraph.models.endpoints.openai_direct import (
    assemble_client_kwargs,
    validate_reasoning_effort,
)
from chemgraph.models.protocols import openai_compatible
from chemgraph.models.supported_models import (
    ARGO_DEFAULT_BASE_URL,
    supported_argo_models,
)
from chemgraph.utils.config_utils import normalize_openai_base_url
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "openai_compatible"

ARGO_PREFIX = "argo:"

# Maps user-facing ``argo:`` model names to the internal wire names expected by
# the Argo API (https://apps.inside.anl.gov/argoapi). When a different endpoint
# (e.g. ArgoProxy) is used, the ``argo:`` prefix is stripped instead and the
# remainder is sent as-is.
ARGO_MODEL_MAP = {
    # GPT family
    "argo:gpt-4o": "gpt4o",
    "argo:gpt-4.1": "gpt41",
    "argo:gpt-4.1-mini": "gpt41mini",
    "argo:gpt-4.1-nano": "gpt41nano",
    "argo:gpt-5": "gpt5",
    "argo:gpt-5-mini": "gpt5mini",
    "argo:gpt-5-nano": "gpt5nano",
    "argo:gpt-5.1": "gpt51",
    "argo:gpt-5.2": "gpt52",
    "argo:gpt-5.4": "gpt54",
    "argo:gpt-5.4-mini": "gpt54mini",
    "argo:gpt-5.4-nano": "gpt54nano",
    "argo:gpt-5.5": "gpt55",
    "argo:gpt-5.6-luna": "gpt56luna",
    "argo:gpt-5.6-sol": "gpt56sol",
    "argo:gpt-5.6-terra": "gpt56terra",
    # Reasoning / o-series
    "argo:o1": "gpto1",
    "argo:o3-mini": "gpto3mini",
    "argo:o3": "gpto3",
    "argo:o4-mini": "gpto4mini",
    # Gemini via Argo
    "argo:gemini-2.5-pro": "gemini25pro",
    "argo:gemini-2.5-flash": "gemini25flash",
    "argo:gemini-3.1-flash-lite": "gemini31flashlite",
    "argo:gemini-3.5-flash": "gemini35flash",
    # Claude via Argo
    "argo:claude-opus-4.6": "claudeopus46",
    "argo:claude-opus-4.5": "claudeopus45",
    "argo:claude-opus-4.1": "claudeopus41",
    "argo:claude-haiku-4.5": "claudehaiku45",
    "argo:claude-sonnet-5": "claudesonnet5",
    "argo:claude-sonnet-4.6": "claudesonnet46",
    "argo:claude-sonnet-4.5": "claudesonnet45",
}

ARGO_LOCAL_OPENAI_MODEL_MAP = {
    # argo-shim advertises GPT-5.4 with this casing. Lowercase gpt-5.4 is
    # rejected by the upstream Argo API behind the shim.
    "argo:gpt-5.4": "GPT-5.4",
}

# Argo authenticates via the 'user' field, not an API key. A non-empty
# placeholder is still required because ChatOpenAI demands a value.
ARGO_CREDENTIAL = CredentialPolicy(env_var="OPENAI_API_KEY", required=False)


def _normalize_argo_model(model_name: str, base_url: str | None) -> str:
    """Normalize an ``argo:``-prefixed model name for the target endpoint.

    * Hosted Argo API endpoints use internal wire names via ``ARGO_MODEL_MAP``.
    * Argo shim, ArgoProxy, and custom OpenAI-compatible endpoints strip the
      ``argo:`` prefix and keep the OpenAI-style name.
    """
    if not model_name.startswith(ARGO_PREFIX):
        return model_name

    model_format = os.getenv("CHEMGRAPH_ARGO_MODEL_FORMAT", "").lower()
    if model_format == "shim":
        return _normalize_argo_local_openai_model(model_name)
    if model_format in {"openai", "openai-compatible"}:
        stripped = model_name.removeprefix(ARGO_PREFIX)
        logger.info("Stripped argo: prefix '%s' -> '%s'", model_name, stripped)
        return stripped
    if model_format in {"wire", "argo"}:
        return _normalize_argo_wire_model(model_name)

    if is_local_http_endpoint(base_url):
        stripped = _normalize_argo_local_openai_model(model_name)
        logger.info(
            "Using OpenAI-style Argo model for local endpoint '%s': '%s' -> '%s'",
            base_url,
            model_name,
            stripped,
        )
        return stripped

    if base_url and "argoapi" in base_url:
        return _normalize_argo_wire_model(model_name)
    else:
        # Non-Argo-API endpoint -- strip prefix only
        stripped = model_name.removeprefix(ARGO_PREFIX)
        logger.info("Stripped argo: prefix '%s' -> '%s'", model_name, stripped)
        return stripped


def _normalize_argo_local_openai_model(model_name: str) -> str:
    """Return the model name expected by local OpenAI-compatible Argo shims."""
    return ARGO_LOCAL_OPENAI_MODEL_MAP.get(
        model_name,
        model_name.removeprefix(ARGO_PREFIX),
    )


def _normalize_argo_wire_model(model_name: str) -> str:
    """Return the hosted-Argo wire model for an ``argo:`` model name."""
    normalized = ARGO_MODEL_MAP.get(model_name)
    if normalized:
        logger.info("Normalized Argo model '%s' -> '%s'", model_name, normalized)
        return normalized

    fallback = model_name.removeprefix(ARGO_PREFIX).replace("-", "").replace(".", "")
    logger.info("Normalized Argo model '%s' -> '%s' (fallback)", model_name, fallback)
    return fallback


def _resolve_argo_api_key(request: ModelRequest) -> str:
    """Resolve the value passed as the ChatOpenAI api_key for Argo routes.

    Mirrors the previous ``load_openai_model``: explicit key, else
    ``OPENAI_API_KEY``, else the argo user / ``ARGO_USER`` / ``"chemgraph"``
    placeholder.
    """
    api_key = request.api_key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = request.argo_user or os.getenv("ARGO_USER", "chemgraph")
    return api_key


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare an ``argo:`` model for hosted, shim, or custom endpoints."""
    requested_model_name = request.model
    base_url = normalize_openai_base_url(request.base_url)

    validate_reasoning_effort(requested_model_name, request.reasoning_effort)

    # Apply default Argo base URL for argo: models when none is specified.
    if not base_url:
        base_url = ARGO_DEFAULT_BASE_URL
        logger.info("Using default Argo base URL: %s", base_url)

    api_key = _resolve_argo_api_key(request)

    if requested_model_name not in supported_argo_models:
        raise ValueError(
            f"Unsupported model '{requested_model_name}'. "
            f"Supported models are: {supported_argo_models}."
        )

    is_argo_claude_model = requested_model_name.startswith("argo:claude-")
    is_argo_endpoint = bool(base_url and "argoapi" in base_url)
    argo_user = (
        request.argo_user or os.getenv("ARGO_USER", "chemgraph")
        if is_argo_endpoint
        else None
    )

    logger.info("Using custom base URL: %s", base_url)
    wire_model = _normalize_argo_model(requested_model_name, base_url)
    if is_argo_endpoint and argo_user:
        logger.info("Using Argo user from config/ARGO_USER/default: %s", argo_user)

    client_kwargs = assemble_client_kwargs(
        model=wire_model,
        requested_model_name=requested_model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=request.temperature,
        reasoning_effort=request.reasoning_effort,
        streaming=is_argo_claude_model,
        argo_user_for_model_kwargs=argo_user if is_argo_endpoint else None,
    )
    return PreparedModel(
        endpoint_name="argo",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        reasoning_effort=request.reasoning_effort,
        supports_structured_output=False,
    )


SPEC = EndpointSpec(
    name="argo",
    protocol=PROTOCOL,
    matches=lambda model: model.startswith(ARGO_PREFIX),
    prepare=prepare,
    protocol_build=openai_compatible.build,
    credential=ARGO_CREDENTIAL,
)
