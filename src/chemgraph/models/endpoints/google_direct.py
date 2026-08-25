"""Direct Gemini endpoint (Google-native protocol).

Logic moved verbatim from ``models/gemini.py``: ``GEMINI_API_KEY`` with
interactive prompt, curated model validation, ``max_output_tokens=6000``.
"""

from __future__ import annotations

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
    resolve_api_key,
)
from chemgraph.models.protocols import google_native
from chemgraph.models.supported_models import supported_gemini_models
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "google_native"

GOOGLE_CREDENTIAL = CredentialPolicy(
    env_var="GEMINI_API_KEY",
    required=True,
    interactive_prompt=True,
)


def resolve_base_url(_model: str, base_url: str | None) -> str | None:
    """Return an optional Google API URL unchanged."""
    return base_url


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a curated Gemini model."""
    model_name = request.model
    api_key = resolve_api_key(GOOGLE_CREDENTIAL, request.api_key)

    if model_name not in supported_gemini_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models are: {supported_gemini_models}."
        )

    logger.info("Loading Gemini model: %s", model_name)
    client_kwargs = dict(
        model=model_name,
        temperature=request.temperature,
        api_key=api_key,
        max_output_tokens=6000,
    )
    return PreparedModel(
        endpoint_name="google_direct",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="google_direct",
    protocol=PROTOCOL,
    matches=lambda model: model in supported_gemini_models,
    prepare=prepare,
    protocol_build=google_native.build,
    credential=GOOGLE_CREDENTIAL,
    config_section="google",
    base_url_resolver=resolve_base_url,
    curated_models=tuple(supported_gemini_models),
    display_name="Google",
)
