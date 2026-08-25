"""Codex endpoint (own-client protocol).

Codex keeps its existing client (``load_codex_model``), which owns ``codex:``
prefix stripping and ChatGPT subscription authentication. The model is immutable
once created, so no temperature or sampling parameters are forwarded.
"""

from __future__ import annotations

from chemgraph.models.codex import CODEX_MODEL_PREFIX
from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
)
from chemgraph.models.protocols import codex_native

PROTOCOL = "codex"

# Subscription-backed; authentication is validated inside the loader.
CODEX_CREDENTIAL = CredentialPolicy(required=False)


def prepare(request: ModelRequest) -> PreparedModel:
    """Prepare a ``codex:`` model. Authentication stays inside the loader."""
    client_kwargs = dict(model_name=request.model)
    return PreparedModel(
        endpoint_name="codex",
        protocol=PROTOCOL,
        client_kwargs=client_kwargs,
        supports_structured_output=True,
    )


SPEC = EndpointSpec(
    name="codex",
    protocol=PROTOCOL,
    matches=lambda model: model.startswith(CODEX_MODEL_PREFIX),
    prepare=prepare,
    protocol_build=codex_native.build,
    credential=CODEX_CREDENTIAL,
    accepted_prefix="codex:",
    display_name="Codex / ChatGPT",
    model_type="Experimental",
)
