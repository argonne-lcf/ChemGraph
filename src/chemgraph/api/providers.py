"""Noninteractive web provider validation and shared credential resolution."""

from dataclasses import dataclass
import hashlib
import json
import os

from chemgraph.api.settings import ConfigurationError, Provider, validate_url


@dataclass(frozen=True)
class SharedCredentials:
    # Never serialize this object into API resources or worker arguments.
    api_key: str | None = None
    argo_user: str | None = None


def endpoint(provider: Provider):
    from chemgraph.models.endpoints import ModelRequest
    from chemgraph.models.endpoints.registry import select_endpoint

    try:
        spec = select_endpoint(
            ModelRequest(model=provider.model, base_url=provider.base_url)
        )
        if spec.name == "codex":
            raise ValueError(
                "Subscription authentication is outside the web deployment."
            )
        if (
            spec.accepted_prefix
            and not provider.model.removeprefix(spec.accepted_prefix).strip()
        ):
            raise ValueError("Missing model name.")
        # Prefix routing alone does not validate Argo/ALCF catalog membership.
        if spec.name.startswith("argo_") or spec.name == "alcf":
            if provider.model not in spec.curated_models:
                raise ValueError("Unsupported institutional model.")
        url = spec.resolve_base_url(provider.model, provider.base_url)
        if url is not None:
            validate_url(url)
        if spec.name.startswith("argo_") and provider.api_key_env:
            raise ValueError("Argo uses argo_user/ARGO_USER, not an API key variable.")
        if spec.name == "ollama" and provider.api_key_env:
            raise ValueError("The Ollama adapter does not support API key overrides.")
        return spec
    except ValueError:
        raise ConfigurationError(
            "Invalid provider route, model, URL, or credential fields."
        ) from None


def credentials(provider: Provider) -> SharedCredentials:
    """Resolve shared credentials independently of authenticated session ownership."""
    spec = endpoint(provider)
    if spec.name.startswith("argo_"):
        identity = provider.argo_user or os.getenv("ARGO_USER", "").strip()
        if not identity or any(c.isspace() for c in identity):
            raise ValueError("missing_identity")
        return SharedCredentials(argo_user=identity)
    variable = provider.api_key_env or spec.credential.env_var
    key = os.getenv(variable, "").strip() if variable else None
    if not key and (provider.api_key_env or spec.credential.required):
        raise ValueError("missing_credentials")
    if spec.name == "vllm":
        # Explicit placeholder prevents the legacy OPENAI_API_KEY fallback.
        key = key or spec.credential.placeholder
    return SharedCredentials(api_key=key)


def status(provider: Provider) -> dict:
    try:
        credentials(provider)
    except ConfigurationError:
        raise
    except ValueError as exc:
        code = str(exc)
        return {
            "configured": False,
            "code": code,
            "message": "The shared Argo identity is missing. Contact your administrator."
            if code == "missing_identity"
            else "Provider credentials are missing. Contact your administrator.",
        }
    return {
        "configured": True,
        "code": "configured",
        "message": "Configured; connectivity has not been verified.",
    }


def model_status(settings) -> dict:
    if settings.demo:
        return {
            "demo": {
                "configured": True,
                "code": "demo",
                "message": "Local demonstration; no LLM is contacted.",
            }
        }
    return {label: status(provider) for label, provider in settings.providers.items()}


def provenance(settings, label: str) -> str | None:
    if settings.demo:
        return "demo" if label == "demo" else None
    if label not in settings.providers:
        return None
    provider = settings.providers[label]
    spec = endpoint(provider)
    payload = (
        spec.name,
        provider.model,
        spec.resolve_base_url(provider.model, provider.base_url),
    )
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def session_status(settings, session) -> dict:
    available = model_status(settings).get(session["model"])
    if available is None:
        return {
            "configured": False,
            "code": "model_removed",
            "message": "This model is no longer enabled. Start a new conversation.",
        }
    if session.get("provenance") != provenance(settings, session["model"]):
        return {
            "configured": False,
            "code": "model_changed",
            "message": "This conversation's model configuration changed. Start a new conversation.",
        }
    return available
