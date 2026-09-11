"""Administrator-owned configuration for the web service."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Provider(BaseModel):
    """An approved model endpoint; secrets are read only by workers."""

    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    argo_user: str | None = None


class Settings(BaseModel):
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

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.demo and not self.dev_user:
            raise ValueError("Demo execution requires an explicit development user.")
        if not self.public_origin.startswith(("http://", "https://")):
            raise ValueError("public_origin must be an HTTP(S) origin.")
        allowed = {"emt", "mace_mp", "mace_off", "mace_polar"}
        if not self.calculators or not set(self.calculators) <= allowed:
            raise ValueError(
                "Only EMT and approved MACE foundation calculators are supported."
            )
        return self

    @classmethod
    def from_env(cls):
        """Read non-secret deployment settings; never expose provider definitions."""
        values = {}
        for field in cls.model_fields:
            value = os.getenv(f"CHEMGRAPH_WEB_{field.upper()}")
            if value is not None:
                values[field] = (
                    json.loads(value)
                    if field in {"providers", "calculators"}
                    else value
                )
        return cls(**values)
