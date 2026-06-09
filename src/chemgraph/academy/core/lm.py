from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class LLMSettings:
    """Configuration for an OpenAI-compatible chat-completions endpoint."""

    base_url: str
    model: str
    provider: str
    timeout_s: float
    temperature: float
    max_tokens: int
    max_retries: int
    retry_delay_s: float
    api_key: str | None = None
    user: str | None = None


def load_lm_config(path: str | Path) -> LLMSettings:
    """Load LM settings from a JSON config file."""
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"LM config must be a JSON object: {config_path}")
    return _settings_from_mapping(data, source=str(config_path))


def _settings_from_mapping(data: dict[str, Any], *, source: str) -> LLMSettings:
    required = (
        "base_url",
        "model",
        "provider",
        "timeout_s",
        "temperature",
        "max_tokens",
        "max_retries",
        "retry_delay_s",
    )
    missing = [name for name in required if data.get(name) is None]
    if missing:
        raise ValueError(f"LM config {source} is missing required keys: {missing}")

    provider = str(data["provider"])
    if provider != "openai_compatible_tools":
        raise ValueError(
            f"LM config {source} provider must be 'openai_compatible_tools'",
        )
    if not data.get("api_key"):
        raise ValueError(
            f"LM config {source} requires api_key; use 'dummy' for Argo shim "
            "routes that do not require auth",
        )

    return LLMSettings(
        base_url=str(data["base_url"]),
        model=str(data["model"]),
        provider=provider,
        api_key=str(data["api_key"]),
        user=str(data["user"]) if data.get("user") else None,
        timeout_s=float(data["timeout_s"]),
        temperature=float(data["temperature"]),
        max_tokens=int(data["max_tokens"]),
        max_retries=int(data["max_retries"]),
        retry_delay_s=float(data["retry_delay_s"]),
    )
