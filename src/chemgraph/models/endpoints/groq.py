"""Groq endpoint (own-client protocol).

Groq keeps its existing client (``load_groq_model``), which owns ``groq:``
prefix stripping, ``GROQ_API_KEY`` handling, and retry-on-auth. This spec routes
``groq:`` models to that loader via the ``groq`` protocol builder.
"""

from __future__ import annotations

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
)
from chemgraph.models.protocols import groq_native

PROTOCOL = "groq"

GROQ_PREFIX = "groq:"

# The loader resolves GROQ_API_KEY itself (with an interactive prompt); the
# policy is declarative metadata for CLI/UI key checks.
GROQ_CREDENTIAL = CredentialPolicy(
    env_var="GROQ_API_KEY",
    required=True,
    interactive_prompt=True,
)


def resolve_base_url(_model: str, base_url: str | None) -> str | None:
    """Return an optional Groq-compatible API URL unchanged."""
    return base_url


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a ``groq:`` model. Key handling stays inside the loader."""
    client_kwargs = dict(
        model_name=request.model,
        temperature=request.temperature,
    )
    if request.api_key is not None:
        client_kwargs["api_key"] = request.api_key
    if request.base_url is not None:
        client_kwargs["base_url"] = request.base_url

    return PreparedModel(
        endpoint_name="groq",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="groq",
    protocol=PROTOCOL,
    matches=lambda model: model.startswith(GROQ_PREFIX),
    prepare=prepare,
    protocol_build=groq_native.build,
    credential=GROQ_CREDENTIAL,
    config_section="groq",
    base_url_resolver=resolve_base_url,
    accepted_prefix=GROQ_PREFIX,
    display_name="GROQ",
)
