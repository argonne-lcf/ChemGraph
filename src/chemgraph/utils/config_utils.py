"""Shared configuration helpers for CLI and UI."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from chemgraph.models.supported_models import (
    all_supported_models,
    supported_anthropic_models,
    supported_argo_models,
    supported_gemini_models,
    supported_ollama_models,
    supported_openai_models,
)


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested TOML-like config into top-level keys used by the CLI."""
    flattened: Dict[str, Any] = {}

    if "general" in config:
        flattened.update(config["general"])

    for section in ["api", "chemistry", "output"]:
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened[f"{section}_{key}_{subkey}"] = subvalue
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


def normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize Argo-style URLs to OpenAI-compatible /v1 URLs."""
    if not base_url:
        return base_url
    if (
        "apps-dev.inside.anl.gov/argoapi" in base_url
        or "apps.inside.anl.gov/argoapi" in base_url
    ):
        base_url = re.sub(r"/api/v1/resource/(chat|embed)/?$", "/v1", base_url)
        base_url = re.sub(r"/docs/?$", "", base_url)
        base_url = re.sub(r"/api/v1/?$", "/v1", base_url)
        base_url = base_url.rstrip("/")
    return base_url


def get_base_url_for_model_from_nested_config(
    model_name: str, config: Dict[str, Any]
) -> Optional[str]:
    """Resolve provider base URL using nested config structure."""
    api = config.get("api", {})

    if model_name in supported_openai_models or model_name in supported_argo_models:
        return normalize_openai_base_url(api.get("openai", {}).get("base_url"))
    if model_name in supported_anthropic_models:
        return api.get("anthropic", {}).get("base_url")
    if model_name in supported_gemini_models:
        return api.get("google", {}).get("base_url")
    if model_name in supported_ollama_models:
        return api.get("local", {}).get("base_url")
    return normalize_openai_base_url(api.get("openai", {}).get("base_url"))


def get_base_url_for_model_from_flat_config(
    model_name: str, config: Dict[str, Any]
) -> Optional[str]:
    """Resolve provider base URL using flattened config keys."""
    if model_name in supported_openai_models or model_name in supported_argo_models:
        return normalize_openai_base_url(config.get("api_openai_base_url"))
    if model_name in supported_anthropic_models:
        return config.get("api_anthropic_base_url")
    if model_name in supported_gemini_models:
        return config.get("api_google_base_url")
    if model_name in supported_ollama_models:
        return config.get("api_local_base_url")
    return normalize_openai_base_url(config.get("api_openai_base_url"))


def get_model_options_for_nested_config(config: Dict[str, Any]) -> list[str]:
    """Return model options based on configured endpoint."""
    base_url = config.get("api", {}).get("openai", {}).get("base_url")
    if base_url and "argoapi" in base_url:
        return supported_argo_models
    return all_supported_models
