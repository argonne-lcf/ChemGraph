"""Direct Anthropic endpoint (Anthropic-native protocol).

Logic moved verbatim from ``models/anthropic.py``: ``ANTHROPIC_API_KEY`` with
interactive prompt, curated model validation, ``max_tokens=6000``.
"""

from __future__ import annotations

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
    resolve_api_key,
)
from chemgraph.models.protocols import anthropic_native
from chemgraph.models.supported_models import supported_anthropic_models
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "anthropic_native"

ANTHROPIC_CREDENTIAL = CredentialPolicy(
    env_var="ANTHROPIC_API_KEY",
    required=True,
    interactive_prompt=True,
)


def resolve_base_url(_model: str, base_url: str | None) -> str | None:
    """Return an optional Anthropic API URL unchanged."""
    return base_url


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a curated Anthropic model."""
    model_name = request.model
    api_key = resolve_api_key(ANTHROPIC_CREDENTIAL, request.api_key)

    if model_name not in supported_anthropic_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models are: {supported_anthropic_models}."
        )

    logger.info("Loading Anthropic model: %s", model_name)
    client_kwargs = dict(
        model=model_name,
        temperature=request.temperature,
        api_key=api_key,
        max_tokens=6000,
    )
    return PreparedModel(
        endpoint_name="anthropic_direct",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="anthropic_direct",
    protocol=PROTOCOL,
    matches=lambda model: model in supported_anthropic_models,
    prepare=prepare,
    protocol_build=anthropic_native.build,
    credential=ANTHROPIC_CREDENTIAL,
    config_section="anthropic",
    base_url_resolver=resolve_base_url,
    curated_models=tuple(supported_anthropic_models),
    display_name="Anthropic",
)
