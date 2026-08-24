"""Ollama endpoint (own-client protocol).

Ollama keeps its existing client (``load_ollama_model``), which validates the
requested model against the curated catalog. Local models need no API key.
"""

from __future__ import annotations

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
)
from chemgraph.models.protocols import ollama_native
from chemgraph.models.supported_models import supported_ollama_models

PROTOCOL = "ollama"

# Local endpoint; no credential required.
OLLAMA_CREDENTIAL = CredentialPolicy(required=False)


def resolve_base_url(_model: str, base_url: str | None) -> str | None:
    """Return an optional local Ollama URL unchanged."""
    return base_url


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a curated Ollama model."""
    client_kwargs = dict(
        model_name=request.model,
        temperature=request.temperature,
    )
    return PreparedModel(
        endpoint_name="ollama",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="ollama",
    protocol=PROTOCOL,
    matches=lambda model: model in supported_ollama_models,
    prepare=prepare,
    protocol_build=ollama_native.build,
    credential=OLLAMA_CREDENTIAL,
    config_section="local",
    base_url_resolver=resolve_base_url,
    curated_models=tuple(supported_ollama_models),
    display_name="Ollama",
    model_type="Local",
)
