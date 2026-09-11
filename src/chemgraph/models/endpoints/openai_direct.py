"""Direct OpenAI endpoint and the shared OpenAI-compatible kwargs assembly.

``assemble_client_kwargs`` is the single place that reproduces the two
parameter shapes the previous ``load_openai_model`` produced -- the
``base_url``-present shape (used by Argo, ALCF, vLLM, custom endpoints) and the
plain OpenAI shape. Every OpenAI-compatible endpoint funnels its kwargs through
it so the protocol builder stays a one-liner.
"""

from __future__ import annotations

from typing import Any

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
    normalize_openai_base_url,
    resolve_api_key,
)
from chemgraph.models.protocols import openai_compatible
from chemgraph.models.supported_models import (
    MODELS_WITHOUT_TEMPERATURE,
    REASONING_EFFORT_CHOICES,
    REASONING_EFFORTS_BY_MODEL,
    supported_openai_models,
)
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "openai_compatible"

OPENAI_CREDENTIAL = CredentialPolicy(
    env_var="OPENAI_API_KEY",
    required=True,
    interactive_prompt=True,
    missing_key_help="OPENAI_API_KEY not set. Please provide an OpenAI API key.",
)


def resolve_base_url(_model: str, base_url: str | None) -> str | None:
    """Normalize an optional direct OpenAI-compatible endpoint URL."""
    return normalize_openai_base_url(base_url)


def validate_reasoning_effort(requested_model_name: str, reasoning_effort: str | None) -> None:
    """Validate reasoning-effort support for a model. Moved from ``openai.py``."""
    if reasoning_effort is None:
        return
    supported_efforts = REASONING_EFFORTS_BY_MODEL.get(requested_model_name)
    if supported_efforts is None:
        raise ValueError(
            f"Model '{requested_model_name}' does not have verified "
            "reasoning-effort support."
        )
    if reasoning_effort not in supported_efforts:
        supported = ", ".join(
            effort for effort in REASONING_EFFORT_CHOICES if effort in supported_efforts
        )
        raise ValueError(
            f"Unsupported reasoning effort '{reasoning_effort}'. "
            f"Model '{requested_model_name}' supports: {supported}."
        )


def assemble_client_kwargs(
    *,
    model: str,
    requested_model_name: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    reasoning_effort: str | None = None,
    streaming: bool = False,
    argo_user_for_model_kwargs: str | None = None,
) -> dict[str, Any]:
    """Build ``ChatOpenAI`` kwargs, mirroring the previous ``load_openai_model``.

    A non-null ``base_url`` selects the custom-endpoint shape (``max_tokens``
    4000 plus sampling params); a null ``base_url`` selects the plain OpenAI
    shape (``max_tokens`` 6000). Models in ``MODELS_WITHOUT_TEMPERATURE`` omit
    all optional sampling parameters.
    """
    minimal = requested_model_name in MODELS_WITHOUT_TEMPERATURE
    if base_url is not None:
        kwargs: dict[str, Any] = dict(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=4000,
        )
        if minimal:
            logger.info(
                "Using minimal request parameters for model '%s'",
                requested_model_name,
            )
        else:
            kwargs.update(
                temperature=temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if streaming:
            kwargs["streaming"] = True
        if argo_user_for_model_kwargs:
            kwargs["model_kwargs"] = {"user": argo_user_for_model_kwargs}
    else:
        kwargs = dict(model=model, api_key=api_key, max_tokens=6000)
        if minimal:
            logger.info(
                "Using minimal request parameters for model '%s'",
                requested_model_name,
            )
        else:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a direct (unprefixed, curated) OpenAI model."""
    requested_model_name = request.model
    base_url = resolve_base_url(requested_model_name, request.base_url)

    validate_reasoning_effort(requested_model_name, request.reasoning_effort)

    if requested_model_name not in supported_openai_models:
        raise ValueError(
            f"Unsupported model '{requested_model_name}'. "
            f"Supported models are: {supported_openai_models}."
        )

    api_key = resolve_api_key(OPENAI_CREDENTIAL, request.api_key)

    logger.info("Loading OpenAI model: %s", requested_model_name)
    client_kwargs = assemble_client_kwargs(
        model=requested_model_name,
        requested_model_name=requested_model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=request.temperature,
        reasoning_effort=request.reasoning_effort,
    )
    return PreparedModel(
        endpoint_name="openai_direct",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        reasoning_effort=request.reasoning_effort,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="openai_direct",
    protocol=PROTOCOL,
    matches=lambda model: model in supported_openai_models,
    prepare=prepare,
    protocol_build=openai_compatible.build,
    credential=OPENAI_CREDENTIAL,
    config_section="openai",
    base_url_resolver=resolve_base_url,
    curated_models=tuple(supported_openai_models),
    display_name="OpenAI",
)
