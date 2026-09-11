"""Administrator-owned configuration for the web service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tomllib
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ConfigurationError(ValueError):
    """A configuration error whose message is safe to display to an operator."""


def validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() for c in value)
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Use an HTTP(S) URL without credentials, query, or fragment.")
    return value.rstrip("/")


class Provider(BaseModel):
    """An approved model endpoint; secrets are read only by workers."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model: str = Field(min_length=1, max_length=200, pattern=r"^\S+$")
    base_url: str | None = None
    api_key_env: str | None = None
    argo_user: str | None = None

    @field_validator("base_url")
    @classmethod
    def endpoint_url(cls, value):
        return validate_url(value) if value is not None else value

    @field_validator("api_key_env")
    @classmethod
    def credential_variable(cls, value):
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("Use an environment variable name.")
        return value

    @field_validator("argo_user")
    @classmethod
    def identity(cls, value):
        if value is not None and (not value.strip() or any(c.isspace() for c in value)):
            raise ValueError("Use a nonblank Argo identity without whitespace.")
        return value


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    data_dir: Path = Path("web_data")
    public_origin: str = "http://localhost:8080"
    identity_header: str = "X-Auth-Request-User"
    dev_user: str | None = None
    demo: bool = False
    providers: dict[str, Provider] = Field(default_factory=dict)
    calculators: tuple[str, ...] = ("emt",)
    max_workers: int = Field(default=2, ge=1, le=32)
    run_timeout: int = Field(default=3600, ge=10)
    upload_limit: int = Field(default=25 * 1024 * 1024, ge=1)
    provider_timeout: int = Field(default=120, ge=1)
    max_queued: int = Field(default=20, ge=1)
    max_user_runs: int = Field(default=2, ge=1)
    clarification_timeout: int = Field(default=900, ge=1)
    user_storage_limit: int = Field(default=1024**3, ge=1)
    retention_days: int = Field(default=30, ge=1)
    context_limit: int = Field(default=16000, ge=1000)

    @field_validator("providers")
    @classmethod
    def provider_labels(cls, value):
        if any(
            not label.strip() or len(label) > 200 or any(ord(c) < 32 for c in label)
            for label in value
        ):
            raise ValueError("Provider labels must contain 1–200 printable characters.")
        return value

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.demo and not self.dev_user:
            raise ValueError("Demo execution requires an explicit development user.")
        self.public_origin = validate_url(self.public_origin)
        if urlsplit(self.public_origin).path:
            raise ValueError("public_origin must not include a path.")
        allowed = {"emt", "mace_mp", "mace_off", "mace_polar"}
        if not self.calculators or not set(self.calculators) <= allowed:
            raise ValueError(
                "Only EMT and approved MACE foundation calculators are supported."
            )
        return self

    @classmethod
    def from_env(cls):
        """Read non-secret deployment settings; never expose provider definitions."""
        try:
            values = {}
            for field in cls.model_fields:
                value = os.getenv(f"CHEMGRAPH_WEB_{field.upper()}")
                if value is not None:
                    values[field] = (
                        json.loads(value)
                        if field in {"providers", "calculators"}
                        else value
                    )
            if "providers" not in values and (
                path := os.getenv("CHEMGRAPH_WEB_PROVIDERS_FILE")
            ):
                with Path(path).open("rb") as source:
                    document = tomllib.load(source)
                if set(document) != {"providers"}:
                    raise ConfigurationError(
                        "The provider file must contain only a providers table."
                    )
                values["providers"] = document["providers"]
            return cls(**values)
        except (OSError, ValueError, ValidationError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            # Parser/validation exception strings may contain literal secrets.
            raise ConfigurationError(
                "Invalid web configuration. Check field names, types, URLs, and provider file syntax/access."
            ) from None
