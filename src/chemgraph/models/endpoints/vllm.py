"""vLLM / custom OpenAI-compatible fallback endpoint.

Consolidates the two ad-hoc fallbacks that previously lived in
``agent/llm_agent.py`` and ``agent/turn.py`` (the latter via
``_custom_openai_compatible_kwargs``). Selected only for an otherwise unknown
model when a base URL is available.

PR 1 preserves the current env-first precedence (``VLLM_BASE_URL`` /
``OPENAI_API_KEY`` win over explicit arguments). The richer ``[api.vllm]``
precedence and deprecation warnings are introduced in PR 3.
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


def resolve_base_url(request: ModelRequest) -> str | None:
    """Resolve the custom endpoint URL (env-first, matching current behavior)."""
    return os.getenv("VLLM_BASE_URL", request.base_url or "") or None


def resolve_api_key(request: ModelRequest) -> str:
    """Resolve the key: OPENAI_API_KEY (deprecated) -> explicit -> dummy."""
    return os.getenv("OPENAI_API_KEY", request.api_key or DUMMY_KEY)


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
)
