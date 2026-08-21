"""ALCF inference endpoints (Sophia default, Minerva, Metis).

Owns cluster URL selection, ``alcf:`` prefix stripping, and the
``ALCF_ACCESS_TOKEN`` credential. Logic moved verbatim from the previous
``models/alcf_endpoints.py``.
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
    ALCF_DEFAULT_BASE_URL,
    ALCF_METIS_BASE_URL,
    ALCF_MINERVA_BASE_URL,
    supported_alcf_metis_models,
    supported_alcf_minerva_models,
    supported_alcf_models,
)
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "openai_compatible"

ALCF_MODEL_PREFIX = "alcf:"

ALCF_MISSING_KEY_HELP = (
    "ALCF access token not found. To authenticate with ALCF:\n"
    "  1. pip install globus_sdk\n"
    "  2. wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/"
    "refs/heads/main/inference_auth_token.py\n"
    "  3. python inference_auth_token.py authenticate\n"
    "  4. export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)\n"
    "\n"
    "See: https://docs.alcf.anl.gov/services/inference-endpoints/#api-access"
)

ALCF_CREDENTIAL = CredentialPolicy(
    env_var="ALCF_ACCESS_TOKEN",
    required=True,
    interactive_prompt=False,
    missing_key_help=ALCF_MISSING_KEY_HELP,
)


def _normalize_alcf_model(model_name: str) -> str:
    """Strip the ``alcf:`` prefix to get the name the endpoint expects."""
    if not model_name.startswith(ALCF_MODEL_PREFIX):
        return model_name
    stripped = model_name.removeprefix(ALCF_MODEL_PREFIX)
    logger.info("Stripped alcf: prefix '%s' -> '%s'", model_name, stripped)
    return stripped


def _resolve_base_url(model_name: str, base_url: str | None) -> str:
    """Pick the cluster base URL that serves ``model_name`` when not given."""
    if base_url:
        return base_url
    if model_name in supported_alcf_minerva_models:
        return ALCF_MINERVA_BASE_URL
    if model_name in supported_alcf_metis_models:
        return ALCF_METIS_BASE_URL
    return ALCF_DEFAULT_BASE_URL


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a curated ``alcf:`` model, selecting its cluster endpoint."""
    requested_model_name = request.model

    # Resolve access token before validating, matching the previous behavior.
    api_key = resolve_api_key(ALCF_CREDENTIAL, request.api_key)

    if requested_model_name not in supported_alcf_models:
        raise ValueError(
            f"Model '{requested_model_name}' is not supported on ALCF. "
            f"Supported models: {supported_alcf_models}"
        )

    base_url = _resolve_base_url(requested_model_name, request.base_url)
    wire_model = _normalize_alcf_model(requested_model_name)

    client_kwargs = dict(model=wire_model, base_url=base_url, api_key=api_key)
    logger.info("Prepared ALCF model: %s from %s", wire_model, base_url)
    return PreparedModel(
        endpoint_name="alcf",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="alcf",
    protocol=PROTOCOL,
    matches=lambda model: model in supported_alcf_models,
    prepare=prepare,
    protocol_build=openai_compatible.build,
    credential=ALCF_CREDENTIAL,
)
