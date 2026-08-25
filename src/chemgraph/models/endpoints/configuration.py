"""Resolve endpoint configuration without constructing model clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemgraph.models.endpoints import EndpointSpec, ModelRequest
from chemgraph.models.endpoints import vllm as vllm_ep
from chemgraph.models.endpoints.registry import match_endpoint, select_endpoint
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _section(api: Mapping[str, Any], name: str | None) -> Mapping[str, Any]:
    if not name:
        return {}
    value = api.get(name)
    return value if isinstance(value, Mapping) else {}


def has_config_section(api: Mapping[str, Any], name: str | None) -> bool:
    """Return whether an endpoint section is a valid mapping."""
    return bool(name and isinstance(api.get(name), Mapping))


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value).strip() or None


def resolve_base_url_for_spec(
    spec: EndpointSpec,
    model: str,
    api: Mapping[str, Any],
    *,
    explicit: str | None = None,
    warn_legacy: bool = True,
) -> str | None:
    """Resolve explicit, canonical, legacy, environment, and default URL values."""
    if explicit:
        return spec.resolve_base_url(model, explicit)

    canonical = spec.config_section
    if has_config_section(api, canonical):
        configured = _text(_section(api, canonical).get("base_url"))
        return spec.resolve_base_url(model, configured, from_config=True)

    if canonical and canonical in api:
        logger.warning(
            "Ignoring malformed [api.%s] section for endpoint '%s'; "
            "expected a table/mapping.",
            canonical,
            spec.name,
        )

    for legacy in spec.legacy_config_sections:
        configured = _text(_section(api, legacy).get("base_url"))
        if configured:
            if warn_legacy:
                logger.warning(
                    "Using legacy [api.%s] base_url for endpoint '%s'; move it to "
                    "[api.%s].base_url.",
                    legacy,
                    spec.name,
                    canonical,
                )
            return spec.resolve_base_url(model, configured, from_config=True)

    return spec.resolve_base_url(model, None, from_config=True)


def select_endpoint_for_config(
    model: str,
    api: Mapping[str, Any],
    *,
    explicit_base_url: str | None = None,
) -> EndpointSpec:
    """Select the model endpoint, consulting vLLM config only as a fallback."""
    spec = match_endpoint(model)
    if spec is not None:
        return spec

    base_url = resolve_base_url_for_spec(
        vllm_ep.SPEC,
        model,
        api,
        explicit=explicit_base_url,
        warn_legacy=False,
    )
    return select_endpoint(ModelRequest(model=model, base_url=base_url))


def resolve_base_url_for_model(
    model: str,
    api: Mapping[str, Any],
    *,
    explicit: str | None = None,
) -> str | None:
    """Select an endpoint and resolve its effective configured base URL."""
    try:
        spec = select_endpoint_for_config(
            model,
            api,
            explicit_base_url=explicit,
        )
    except ValueError:
        return explicit
    return resolve_base_url_for_spec(spec, model, api, explicit=explicit)


def endpoint_api_key(
    spec: EndpointSpec,
    api: Mapping[str, Any],
) -> str | None:
    """Read an API key only from the selected endpoint's canonical section."""
    return _text(_section(api, spec.config_section).get("api_key"))


def resolve_argo_user(
    api: Mapping[str, Any],
    *,
    explicit: Any = None,
) -> str | None:
    """Resolve canonical and one-release legacy Argo user settings."""
    value = _text(explicit)
    if value:
        return value

    canonical = _section(api, "argo")
    value = _text(canonical.get("argo_user"))
    if value:
        return value

    value = _text(canonical.get("user"))
    if value:
        logger.warning(
            "Using compatibility [api.argo].user; rename it to "
            "[api.argo].argo_user."
        )
        return value

    value = _text(_section(api, "openai").get("argo_user"))
    if value:
        logger.warning(
            "Using legacy [api.openai].argo_user; move it to "
            "[api.argo].argo_user."
        )
        return value
    return None
