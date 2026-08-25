from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemgraph.models.endpoints.configuration import (
    endpoint_api_key,
    resolve_argo_user,
    resolve_base_url_for_spec,
    select_endpoint_for_config,
)
from chemgraph.models.endpoints.registry import match_endpoint

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
    reasoning_effort: str | None = None
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
        reasoning_effort: str | None = None,
    ) -> None:
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "argo_user", argo_user or user)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "reasoning_effort", reasoning_effort)
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
        argo_user=_str_or_none(data.get("argo_user") or data.get("user")),
        provider=_str_or_none(provider),
        reasoning_effort=_str_or_none(data.get("reasoning_effort")),
        timeout_s=_float_or_none(data.get("timeout_s")),
        temperature=_float_or_none(data.get("temperature")),
        max_tokens=_int_or_none(data.get("max_tokens")),
        max_retries=_int_or_none(data.get("max_retries")),
        retry_delay_s=_float_or_none(data.get("retry_delay_s")),
    )


def _extract_endpoint_from_cli_toml(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Pull LLM endpoint fields out of the CLI's nested TOML structure."""
    general_value = raw.get("general") or {}
    api_value = raw.get("api") or {}
    general = general_value if isinstance(general_value, Mapping) else {}
    api = api_value if isinstance(api_value, Mapping) else {}
    model = general.get("model")
    provider = general.get("provider") or raw.get("provider")
    explicit_base_url = _str_or_none(
        general.get("base_url") or raw.get("base_url")
    )
    explicit_api_key = _str_or_none(general.get("api_key") or raw.get("api_key"))

    spec = None
    if isinstance(model, str):
        try:
            spec = select_endpoint_for_config(
                model,
                api,
                explicit_base_url=explicit_base_url,
            )
        except ValueError:
            pass

    base_url = explicit_base_url
    api_key = explicit_api_key
    if spec is not None and isinstance(model, str):
        base_url = resolve_base_url_for_spec(
            spec,
            model,
            api,
            explicit=explicit_base_url,
        )
        api_key = explicit_api_key or endpoint_api_key(spec, api)

    return {
        "model": model,
        "base_url": base_url,
        "argo_user": resolve_argo_user(
            api,
            explicit=general.get("argo_user") or general.get("user"),
        ),
        "api_key": api_key,
        "provider": provider,
        "reasoning_effort": general.get("reasoning_effort"),
    }


def _provider_section_for(model: Any) -> str:
    """Return the canonical config section for a model identifier."""
    if isinstance(model, str):
        spec = match_endpoint(model)
        if spec is not None and spec.config_section:
            return spec.config_section
    return "vllm"


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
