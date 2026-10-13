"""Explicit, immutable runtime configuration. Secrets are environment references."""

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Principal(StrictModel):
    name: str = Field(min_length=1)
    token_env: str
    role: Literal["peer", "operator", "notifier"] = "peer"


class Calculation(StrictModel):
    name: str
    arguments: dict


class HPC(StrictModel):
    """Operator-authored, fixed submission templates; never agent-selected endpoints."""

    backend: Literal["globus_compute"] = "globus_compute"
    server_url: str
    token_env: str | None = None
    endpoint_id: str
    tool: str
    calculator: str
    model: str
    scheduler_limits: str = Field(min_length=1)
    calculations: tuple[Calculation, ...]

    @model_validator(mode="after")
    def validate_templates(self):
        url = urlparse(self.server_url)
        if url.scheme not in {"https", "http"} or not url.hostname:
            raise ValueError("MCP server must be a configured HTTP(S) URL")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote MCP requires HTTPS")
        if url.username or url.password:
            raise ValueError("MCP credentials must use token_env")
        names = [c.name for c in self.calculations]
        if not names or len(names) != len(set(names)):
            raise ValueError("Configure distinct fixed calculation templates")
        return self


class Profile(StrictModel):
    version: Literal[3] = 3
    adapter: Literal["chemgraph"] = "chemgraph"
    model: str = "gpt-4o-mini"
    recursion_limit: int = Field(default=40, ge=1, le=200)
    max_submissions: int = Field(default=5, ge=1, le=100)
    model_roots: tuple[Path, ...] = ()
    capabilities: tuple[str, ...] = (
        "run_ase", "file_to_atomsdata", "save_atomsdata_to_file", "smiles_to_atomsdata",
        "smiles_to_coordinate_file", "extract_output_json", "get_symmetry_number", "is_linear_molecule",
    )
    calculators: tuple[str, ...] = (
        "emt", "mace_polar", "mace_mp", "mace_off", "mace_anicc", "fairchem",
        "aimnet2", "tblite", "nwchem", "orca",
    )
    default_calculator: dict | None = None
    model_resources: dict = Field(default_factory=dict)
    executable_bindings: dict = Field(default_factory=dict)
    limits: dict = Field(default_factory=dict)
    hpc: HPC | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_catalog(cls, value):
        if isinstance(value, dict) and (
            "tools" in value or (isinstance(value.get("version"), int) and value["version"] < 3)
        ):
            raise ValueError(
                "Use profile version 3 and remove profile.tools; select capabilities. "
                "Start with fresh service state; existing checkpoints require their original runtime."
            )
        return value

    @field_validator("model_roots")
    @classmethod
    def absolute_model_roots(cls, roots):
        if any(not root.is_absolute() for root in roots):
            raise ValueError("model_roots must be absolute operator-controlled directories")
        return tuple(root.resolve() for root in roots)

    @property
    def identity(self) -> str:
        from importlib.metadata import version
        from chemgraph.execution.scoped import EXECUTION_CONTRACT_VERSION

        value = json.dumps(
            {
                **self.model_dump(mode="json"),
                "chemgraph_version": version("chemgraph"),
                "execution_contract": EXECUTION_CONTRACT_VERSION,
            },
            sort_keys=True,
        ).encode()
        return "service-v3-" + hashlib.sha256(value).hexdigest()

    def execution_policy(self):
        from chemgraph.execution.prepared import ExecutionPolicy, FRAMEWORK_HELPERS

        return ExecutionPolicy(
            tools=frozenset(self.capabilities), calculators=frozenset(self.calculators),
            helpers=FRAMEWORK_HELPERS | ({"submit_hpc"} if self.hpc else set()),
            configuration_id=self.identity, default_calculator=self.default_calculator,
            model_resources=self.model_resources, executable_bindings=self.executable_bindings,
            limits=self.limits,
        )

    @model_validator(mode="after")
    def validate_capabilities(self):
        from chemgraph.execution.prepared import ASE_TOOLS, CALCULATORS

        if set(self.capabilities) - ASE_TOOLS or set(self.calculators) - CALCULATORS:
            raise ValueError("Configure only supported prepared ASE capabilities/calculators")
        self.execution_policy()
        return self


class Settings(StrictModel):
    state_dir: Path
    public_url: str = "http://127.0.0.1:8080"
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    principals: tuple[Principal, ...]
    profile: Profile = Profile()

    @model_validator(mode="after")
    def validate_config(self):
        parsed = urlparse(self.public_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("public_url must be an HTTP(S) origin")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("public_url must be an origin without credentials or a path")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote A2A requires an HTTPS public_url")
        names = [p.name for p in self.principals]
        if not names or len(names) != len(set(names)):
            raise ValueError("Configure distinct authenticated principal names")
        return self

    def credentials(self) -> dict[str, Principal]:
        result = {}
        for principal in self.principals:
            token = os.environ.get(principal.token_env, "")
            if len(token) < 32:
                raise ValueError(f"{principal.token_env} must contain at least 32 characters")
            if token in result:
                raise ValueError("Each principal needs a distinct token")
            result[token] = principal
        return result


def load_settings(path: str | Path) -> Settings:
    path = Path(path).resolve()
    data = tomllib.loads(path.read_text())
    state = Path(data["state_dir"]).expanduser()
    data["state_dir"] = (path.parent / state).resolve()
    return Settings.model_validate(data)
