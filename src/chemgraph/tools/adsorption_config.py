"""Runtime configuration for external adsorption engines."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveFloat


class AdsorptionRuntimeConfig(BaseModel):
    """Deployment-only settings serialized to execution workers."""

    engine: Literal["graspa_sycl", "graspa_cuda"]
    executable: str = Field(min_length=1)
    timeout_seconds: PositiveFloat = 7200.0
    environment: dict[str, str] = Field(default_factory=dict)


def _find_config(config_path: str | None) -> Path | None:
    if config_path:
        return Path(config_path).expanduser()
    env_path = os.getenv("CHEMGRAPH_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    candidate = Path.cwd() / "config.toml"
    if candidate.is_file():
        return candidate
    candidate = Path(__file__).resolve().parents[3] / "config.toml"
    return candidate if candidate.is_file() else None


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_adsorption_runtime(
    config_path: str | None = None,
) -> AdsorptionRuntimeConfig:
    """Load one active adsorption engine from ChemGraph TOML."""

    path = _find_config(config_path)
    if path is None or not path.is_file():
        raise RuntimeError(
            "Adsorption runtime is not configured. Add [adsorption] to "
            "config.toml or set CHEMGRAPH_CONFIG."
        )
    data = _load_toml(path)
    canonical = data.get("adsorption")
    legacy = data.get("graspa")
    if canonical is not None and legacy is not None:
        raise ValueError("Configure [adsorption] or legacy [graspa], not both")
    if canonical is not None:
        return AdsorptionRuntimeConfig.model_validate(canonical)
    if legacy is not None:
        warnings.warn(
            "[graspa] is deprecated; use [adsorption] with engine=graspa_*",
            DeprecationWarning,
            stacklevel=2,
        )
        mapped = dict(legacy)
        runtime = str(mapped.pop("runtime", "sycl")).lower()
        mapped["engine"] = {
            "sycl": "graspa_sycl",
            "cuda": "graspa_cuda",
        }.get(runtime, runtime)
        return AdsorptionRuntimeConfig.model_validate(mapped)
    raise RuntimeError(f"No [adsorption] section found in {path}")
