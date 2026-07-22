"""LM endpoint settings, owned by swarm.

Historically vendored here (instead of imported from
``chemgraph.models.settings``) so the eventual repo split would be
mechanical. That split happened 2026-07-07; this module is now
canonical.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclasses.dataclass(frozen=True, init=False)
class LLMSettings:
    """Fully resolved description of one LLM endpoint."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    argo_user: str | None = None
    provider: str | None = None
    timeout_s: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    retry_delay_s: float | None = None

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        argo_user: str | None = None,
        provider: str | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        retry_delay_s: float | None = None,
        user: str | None = None,
    ) -> None:
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "argo_user", argo_user or user)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "max_retries", max_retries)
        object.__setattr__(self, "retry_delay_s", retry_delay_s)

    @property
    def user(self) -> str | None:
        """Backward-compatible academy name for Argo user metadata."""
        return self.argo_user


def load_lm_settings(source: str | Path | Mapping[str, Any]) -> LLMSettings:
    """Build LLMSettings from a JSON file, TOML file, or already-parsed dict."""
    if isinstance(source, Mapping):
        return _from_mapping(source)

    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".toml":
        raw = tomllib.loads(text)
        return _from_mapping(_extract_endpoint_from_cli_toml(raw))
    return _from_mapping(json.loads(text))


def _from_mapping(data: Mapping[str, Any]) -> LLMSettings:
    if not isinstance(data, Mapping):
        raise ValueError("LM config must be a mapping/object")

    model = data.get("model") or data.get("model_name")
    if not isinstance(model, str) or not model:
        raise ValueError("LM config requires a non-empty 'model' field")

    provider = data.get("provider")
    if provider is not None and provider != "openai_compatible_tools":
        raise ValueError(
            "LM config 'provider' must be 'openai_compatible_tools' or absent",
        )

    api_key = data.get("api_key")
    if provider == "openai_compatible_tools" and not api_key:
        raise ValueError(
            "openai_compatible_tools provider requires api_key "
            "(use 'dummy' for Argo shim routes that ignore auth)",
        )

    return LLMSettings(
        model=str(model),
        base_url=_str_or_none(data.get("base_url")),
        api_key=_str_or_none(api_key),
        argo_user=_str_or_none(data.get("user") or data.get("argo_user")),
        provider=_str_or_none(provider),
        timeout_s=_float_or_none(data.get("timeout_s")),
        temperature=_float_or_none(data.get("temperature")),
        max_tokens=_int_or_none(data.get("max_tokens")),
        max_retries=_int_or_none(data.get("max_retries")),
        retry_delay_s=_float_or_none(data.get("retry_delay_s")),
    )


def _extract_endpoint_from_cli_toml(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Pull LLM endpoint fields out of the CLI's nested TOML structure."""
    general = raw.get("general") or {}
    api = raw.get("api") or {}
    model = general.get("model")
    argo_user = general.get("argo_user") or (api.get("argo") or {}).get("user")

    base_url = None
    if isinstance(model, str):
        if model.startswith("argo:"):
            base_url = (api.get("argo") or {}).get("base_url")
        else:
            for section_name in ("openai", "anthropic", "gemini", "alcf", "ollama"):
                section = api.get(section_name) or {}
                if section.get("base_url"):
                    base_url = section["base_url"]
                    break

    return {
        "model": model,
        "base_url": base_url,
        "argo_user": argo_user,
        "api_key": (api.get(_provider_section_for(model)) or {}).get("api_key"),
    }


def _provider_section_for(model: Any) -> str:
    if isinstance(model, str):
        if model.startswith("argo:"):
            return "argo"
        if model.startswith("groq:"):
            return "groq"
    return "openai"


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value) or None


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)
