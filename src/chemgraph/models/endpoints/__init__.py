"""Endpoint specifications for the model-loading pipeline.

Each endpoint module declares what makes it different from other endpoints that
speak the same protocol. The loader selects one ``EndpointSpec`` per request.
"""

from chemgraph.models.endpoints.base import (
    CredentialPolicy,
    EndpointSpec,
    ModelRequest,
    PreparedModel,
    is_local_http_endpoint,
    resolve_api_key,
)

__all__ = [
    "CredentialPolicy",
    "EndpointSpec",
    "ModelRequest",
    "PreparedModel",
    "is_local_http_endpoint",
    "resolve_api_key",
]
