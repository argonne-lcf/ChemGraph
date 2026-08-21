"""Aurora on-node inference endpoint (OpenAI-compatible).

Owns ``aurora:`` prefix stripping, the ``AURORA_API_KEY`` credential (with an
``OPENAI_API_KEY`` fallback and a ``"dummy"`` placeholder for on-node servers
that do not enforce auth), and base URL resolution against
``AURORA_BASE_URL`` / ``AURORA_DEFAULT_BASE_URL``. Logic moved verbatim from
``models/aurora_endpoints.py``.
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
from chemgraph.models.supported_models import AURORA_DEFAULT_BASE_URL
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROTOCOL = "openai_compatible"

AURORA_PREFIX = "aurora:"

DUMMY_KEY = "dummy"

# Aurora on-node servers (llama-server / vLLM-XPU) typically do not enforce
# authentication, so a placeholder key is accepted. The deprecated
# OPENAI_API_KEY fallback is applied explicitly in ``resolve_api_key`` below.
AURORA_CREDENTIAL = CredentialPolicy(
    env_var="AURORA_API_KEY",
    required=False,
    placeholder=DUMMY_KEY,
)


def resolve_base_url(request: ModelRequest) -> str:
    """Resolve the Aurora endpoint URL.

    Order: explicit ``request.base_url`` -> ``AURORA_BASE_URL`` env ->
    ``AURORA_DEFAULT_BASE_URL`` (a co-located ``127.0.0.1`` server).
    """
    if request.base_url:
        return request.base_url
    return os.getenv("AURORA_BASE_URL") or AURORA_DEFAULT_BASE_URL


def resolve_api_key(request: ModelRequest) -> str:
    """Resolve the API key.

    Order: explicit ``request.api_key`` -> ``AURORA_API_KEY`` env ->
    ``OPENAI_API_KEY`` env -> ``"dummy"`` placeholder. ``langchain_openai``
    requires a non-empty value, so we never return ``None``.
    """
    if request.api_key:
        return request.api_key
    return (
        os.getenv("AURORA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or DUMMY_KEY
    )


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare an ``aurora:`` model for the OpenAI-compatible protocol.

    The ``aurora:`` prefix is stripped before the name is sent to the endpoint;
    what remains must match the server's advertised model id
    (``llama-server --alias`` / ``--served-model-name``).
    """
    wire_model = request.model.removeprefix(AURORA_PREFIX)
    base_url = resolve_base_url(request)
    api_key = resolve_api_key(request)

    logger.info("Loading Aurora model: %s from %s", wire_model, base_url)
    client_kwargs = dict(
        model=wire_model,
        base_url=base_url,
        api_key=api_key,
        temperature=request.temperature,
    )

    return PreparedModel(
        endpoint_name="aurora",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="aurora",
    protocol=PROTOCOL,
    matches=lambda model: model.startswith(AURORA_PREFIX),
    prepare=prepare,
    protocol_build=openai_compatible.build,
    credential=AURORA_CREDENTIAL,
)
