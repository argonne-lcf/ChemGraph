"""Allowlisted model routing metadata; credentials never enter session identity."""

from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict


class ModelEndpointDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint_name: str
    protocol: str
    requested_model: str
    effective_model: str
    base_url: str | None = None
    # Keep endpoint preparation defaults when a URL originally came from the SDK.
    configured_base_url: bool = False
    caller_owned: bool = False


def public_base_url(value: str | None) -> tuple[str | None, bool]:
    """Return a canonical URL only when it contains no credential-bearing parts."""
    if not value:
        return None, False
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        return None, True
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port and (parsed.scheme, port) not in {("https", 443), ("http", 80)}:
        host += f":{port}"
    return urlunsplit((parsed.scheme.lower(), host, parsed.path.rstrip("/"), "", "")), False


def describe_model_endpoint(prepared, client, model_name, requested_base_url=None):
    """Read resolved routing from the client without serializing its internals."""
    kwargs = prepared.client_kwargs
    url = kwargs.get("base_url") or requested_base_url
    configured = bool(url)
    # OpenAI/Groq expose the actual SDK route, including environment defaults.
    root = getattr(client, "root_client", None)
    if root is not None and getattr(root, "base_url", None) is not None:
        url = str(root.base_url)
    elif prepared.protocol == "anthropic_native":
        url = getattr(client, "anthropic_api_url", None) or url
    elif prepared.protocol == "groq":
        sdk = getattr(getattr(client, "client", None), "_client", None)
        url = str(sdk.base_url) if sdk is not None else getattr(client, "groq_api_base", None) or url
    elif prepared.protocol == "ollama":
        local_client = getattr(client, "_client", None)
        transport = getattr(local_client, "_client", None)
        if transport is not None and getattr(transport, "base_url", None) is not None:
            url = str(transport.base_url)
    safe_url, opaque = public_base_url(url)
    if requested_base_url and prepared.protocol not in {"openai_compatible", "anthropic_native", "groq", "ollama"}:
        safe_url, opaque = None, True
    return ModelEndpointDescriptor(
        endpoint_name=prepared.endpoint_name, protocol=prepared.protocol,
        requested_model=model_name,
        effective_model=kwargs.get("model", kwargs.get("model_name", model_name)),
        base_url=safe_url, configured_base_url=configured, caller_owned=opaque,
    )
