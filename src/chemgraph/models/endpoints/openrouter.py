"""OpenRouter endpoint (OpenAI-compatible).

Owns ``openrouter:`` prefix stripping, the ``OPENROUTER_API_KEY`` credential
(never ``OPENAI_API_KEY``), the higher default token budget, and the restricted
optional-parameter set. Logic moved verbatim from ``models/openrouter.py``.
"""

from __future__ import annotations

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
    resolve_api_key,
)
from chemgraph.models.protocols import openai_compatible
from chemgraph.models.supported_models import (
    MODELS_WITHOUT_TEMPERATURE,
    OPENROUTER_DEFAULT_BASE_URL,
)
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "openai_compatible"

OPENROUTER_PREFIX = "openrouter:"

# Higher than the OpenAI/Groq defaults on purpose: most models served through
# OpenRouter emit reasoning tokens by default, and those count against the
# *completion* budget. Too small a cap lets chain-of-thought exhaust it before
# the tool call is emitted, producing an empty turn the graph reads as a dead end.
OPENROUTER_DEFAULT_MAX_TOKENS = 8000

OPENROUTER_MISSING_KEY_HELP = (
    "OpenRouter API key not found. Set the OPENROUTER_API_KEY "
    "environment variable:\n"
    "  export OPENROUTER_API_KEY='your_key_here'\n"
    "  Get a key at: https://openrouter.ai/keys"
)

# Falls back to OPENROUTER_API_KEY. Never to OPENAI_API_KEY.
OPENROUTER_CREDENTIAL = CredentialPolicy(
    env_var="OPENROUTER_API_KEY",
    required=True,
    interactive_prompt=True,
    missing_key_help=OPENROUTER_MISSING_KEY_HELP,
)


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare an ``openrouter:`` model. The slug is sent verbatim."""
    # Keep the prefixed name -- it is what the quirk sets are keyed on.
    requested_model_name = request.model
    model_name = requested_model_name.removeprefix(OPENROUTER_PREFIX)

    api_key = resolve_api_key(OPENROUTER_CREDENTIAL, request.api_key)

    logger.info("Loading OpenRouter model: %s", model_name)
    client_kwargs = dict(
        model=model_name,
        api_key=api_key,
        base_url=request.base_url or OPENROUTER_DEFAULT_BASE_URL,
        max_tokens=OPENROUTER_DEFAULT_MAX_TOKENS,
    )
    # top_p / frequency_penalty / presence_penalty are deliberately not sent:
    # they are no-ops at temperature 0, and OpenRouter fans requests out to a
    # rotating pool of upstream providers whose real parameter support is
    # narrower than the advertised union.
    if requested_model_name not in MODELS_WITHOUT_TEMPERATURE:
        client_kwargs["temperature"] = request.temperature

    return PreparedModel(
        endpoint_name="openrouter",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="openrouter",
    protocol=PROTOCOL,
    matches=lambda model: model.startswith(OPENROUTER_PREFIX),
    prepare=prepare,
    protocol_build=openai_compatible.build,
    credential=OPENROUTER_CREDENTIAL,
)
