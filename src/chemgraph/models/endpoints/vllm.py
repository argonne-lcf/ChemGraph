"""vLLM / custom OpenAI-compatible fallback endpoint.

Consolidates the two ad-hoc fallbacks that previously lived in
``agent/llm_agent.py`` and ``agent/turn.py`` (the latter via
``_custom_openai_compatible_kwargs``). Selected only for an otherwise unknown
model when a base URL is available.

Configuration and credential precedence is owned here so every caller observes
the same explicit/canonical/environment/legacy behavior.
"""

from __future__ import annotations

import os

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
)
from chemgraph.models.protocols import openai_compatible
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "openai_compatible"

DUMMY_KEY = "dummy_vllm_key"

# The endpoint may not use a key; a placeholder is accepted. The deprecated
# OPENAI_API_KEY fallback is applied explicitly below (the one sanctioned
# cross-provider credential exception).
VLLM_CREDENTIAL = CredentialPolicy(
    env_var="VLLM_API_KEY",
    required=False,
    placeholder=DUMMY_KEY,
)


def resolve_configured_base_url(_model: str, base_url: str | None) -> str | None:
    """Resolve explicit/configured URL before the vLLM environment variable."""
    return base_url or os.getenv("VLLM_BASE_URL") or None


def resolve_base_url(request: ModelRequest) -> str | None:
    """Resolve an explicit/configured URL before the vLLM environment value."""
    return resolve_configured_base_url(request.model, request.base_url)


def resolve_api_key(request: ModelRequest) -> str:
    """Resolve explicit, vLLM, deprecated OpenAI, then placeholder credentials."""
    if request.api_key:
        return request.api_key
    if api_key := os.getenv("VLLM_API_KEY"):
        return api_key
    if api_key := os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "Using deprecated OPENAI_API_KEY for a vLLM/custom endpoint; "
            "set VLLM_API_KEY instead."
        )
        return api_key
    return DUMMY_KEY


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a custom OpenAI-compatible endpoint for an unknown model."""
    base_url = resolve_base_url(request)
    if not base_url:
        raise ValueError(
            f"Unsupported model or missing base URL for: {request.model}"
        )
    api_key = resolve_api_key(request)

    logger.info(
        "Attempting to load model '%s' from custom endpoint: %s",
        request.model,
        base_url,
    )
    client_kwargs = dict(
        model=request.model,
        temperature=request.temperature,
        base_url=base_url,
        api_key=api_key,
        max_tokens=4000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    user = request.argo_user or os.getenv("ARGO_USER")
    if base_url and "argoapi" in base_url and user:
        client_kwargs["model_kwargs"] = {"user": user}

    return PreparedModel(
        endpoint_name="vllm",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


def can_handle(request: ModelRequest) -> bool:
    """Return True when a custom endpoint URL is available for the fallback."""
    return resolve_base_url(request) is not None


SPEC = EndpointSpec(
    name="vllm",
    protocol=PROTOCOL,
    # Matching is resolved by the loader as a last resort, not by model name.
    matches=lambda model: False,
    prepare=prepare,
    protocol_build=openai_compatible.build,
    credential=VLLM_CREDENTIAL,
    config_section="vllm",
    legacy_config_sections=("openai",),
    base_url_resolver=resolve_configured_base_url,
    display_name="vLLM / Custom",
)
