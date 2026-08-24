"""Shared configuration helpers for CLI and UI."""

from __future__ import annotations

from typing import Any, Dict, Optional

from chemgraph.models.endpoints import (
    normalize_openai_base_url as _normalize_openai_base_url,
)
from chemgraph.models.endpoints.configuration import (
    resolve_argo_user,
    resolve_base_url_for_model,
)
from chemgraph.models.endpoints.registry import (
    catalog_entries,
    catalog_models,
    config_sections,
)


def normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    """Backward-compatible wrapper for endpoint URL normalization."""
    return _normalize_openai_base_url(base_url)


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested TOML-like config into top-level keys used by the CLI.

    Parameters
    ----------
    config : dict[str, Any]
        Nested configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Flattened configuration with section names included in keys.
    """
    flattened: Dict[str, Any] = {}

    if "general" in config:
        flattened.update(config["general"])

    for section in ["api", "chemistry", "output"]:
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened[f"{section}_{key}_{subkey}"] = subvalue
                    if section == "api" and key == "vllm" and "base_url" not in value:
                        flattened["api_vllm_base_url"] = ""
                else:
                    flattened[f"{section}_{key}"] = value

    for section in ["logging", "features", "security", "advanced"]:
        if section in config:
            if isinstance(config[section], dict):
                for key, value in config[section].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flattened[f"{section}_{key}_{subkey}"] = subvalue
                    else:
                        flattened[f"{section}_{key}"] = value
            else:
                flattened[section] = config[section]

    return flattened


def _api_from_flat_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Reconstruct endpoint sections while preserving explicit empty values."""
    api: Dict[str, Dict[str, Any]] = {}
    for section in config_sections():
        prefix = f"api_{section}_"
        values = {
            key.removeprefix(prefix): value
            for key, value in config.items()
            if key.startswith(prefix)
        }
        if values:
            api[section] = values
    return api


def get_base_url_for_model_from_nested_config(
    model_name: str, config: Dict[str, Any]
) -> Optional[str]:
    """Resolve provider base URL using nested config structure.

    Parameters
    ----------
    model_name : str
        Model identifier.
    config : dict[str, Any]
        Nested configuration dictionary.

    Returns
    -------
    str or None
        Matching provider base URL, or ``None`` when not configured.
    """
    api_value = config.get("api", {})
    api = api_value if isinstance(api_value, dict) else {}
    return resolve_base_url_for_model(model_name, api)


def get_base_url_for_model_from_flat_config(
    model_name: str, config: Dict[str, Any]
) -> Optional[str]:
    """Resolve provider base URL using flattened config keys.

    Parameters
    ----------
    model_name : str
        Model identifier.
    config : dict[str, Any]
        Flattened configuration dictionary.

    Returns
    -------
    str or None
        Matching provider base URL, or ``None`` when not configured.
    """
    return resolve_base_url_for_model(model_name, _api_from_flat_config(config))


def get_model_options_for_nested_config(config: Dict[str, Any]) -> list[str]:
    """Return model options for UI selection.

    Always show all curated models so users can switch providers from the UI.
    If Argo endpoint is configured, prioritize Argo model IDs at the top.

    Parameters
    ----------
    config : dict[str, Any]
        Nested configuration dictionary.

    Returns
    -------
    list[str]
        Model identifiers for UI selection.
    """
    models = catalog_models()
    argo_models = [
        model
        for model, spec in catalog_entries()
        if spec.config_section == "argo"
    ]
    api = config.get("api", {})
    if not isinstance(api, dict):
        return models

    argo = api.get("argo")
    canonical_url = argo.get("base_url") if isinstance(argo, dict) else None
    openai = api.get("openai")
    legacy_url = openai.get("base_url") if isinstance(openai, dict) else None
    legacy_argo = "argo" not in api and legacy_url and "argoapi" in legacy_url
    if legacy_argo and argo_models:
        resolve_base_url_for_model(argo_models[0], api)

    if canonical_url or legacy_argo:
        remaining = [model for model in models if model not in argo_models]
        return argo_models + remaining
    return models


def get_argo_user_from_nested_config(config: Dict[str, Any]) -> Optional[str]:
    """Resolve Argo user from nested config.

    Parameters
    ----------
    config : dict[str, Any]
        Nested configuration dictionary.

    Returns
    -------
    str or None
        Configured Argo username, or ``None``.
    """
    api = config.get("api", {})
    return resolve_argo_user(api if isinstance(api, dict) else {})


def get_argo_user_from_flat_config(config: Dict[str, Any]) -> Optional[str]:
    """Resolve Argo user from flattened config.

    Parameters
    ----------
    config : dict[str, Any]
        Flattened configuration dictionary.

    Returns
    -------
    str or None
        Configured Argo username, or ``None``.
    """
    return resolve_argo_user(_api_from_flat_config(config))
