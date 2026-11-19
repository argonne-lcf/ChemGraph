"""Operator-controlled settings for the single-host service."""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from chemgraph.runtime.policy import ExecutionPolicy, SCOPED_TOOLS


@dataclass(frozen=True)
class ServiceConfig:
    data_dir: Path
    public_url: str
    host: str = "127.0.0.1"
    port: int = 9003
    allowed_origins: tuple[str, ...] = ()
    max_workers: int = 4
    per_user_running: int = 1
    per_user_queued: int = 10
    max_text_bytes: int = 1024 * 1024
    max_upload_bytes: int = 100 * 1024 * 1024
    max_storage_bytes: int = 10 * 1024 * 1024 * 1024
    max_job_seconds: int = 3600
    shutdown_seconds: int = 30
    max_inline_tools: int = 4
    enable_pubchem: bool = False
    policy: ExecutionPolicy = field(default_factory=lambda: ExecutionPolicy(
        limits={"max_steps": 1000, "max_atoms": 1000}))

    def __post_init__(self):
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        url = urlsplit(self.public_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment or url.path not in {"", "/"}
                or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
            raise ValueError("public_url must be an HTTPS origin (HTTP is allowed only on loopback).")
        for key in ("port", "max_workers", "per_user_running", "per_user_queued", "max_text_bytes",
                    "max_upload_bytes", "max_storage_bytes", "max_job_seconds", "shutdown_seconds",
                    "max_inline_tools"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if not self.policy.tools <= SCOPED_TOOLS | {"molecule_name_to_smiles"}:
            raise ValueError("Unsupported service tool policy.")
        object.__setattr__(self, "allowed_origins", tuple(self.allowed_origins) or (self.origin,))

    @property
    def origin(self):
        url = urlsplit(self.public_url)
        return f"{url.scheme}://{url.netloc}"

    @property
    def allowed_host(self):
        return urlsplit(self.public_url).netloc.lower()

    @property
    def database(self):
        return self.data_dir / "service.sqlite3"


def load_config(path):
    import tomllib

    with open(path, "rb") as stream:
        raw = tomllib.load(stream)
    values = raw.get("service", {})
    policy = raw.get("policy", {})
    if policy:
        values["policy"] = ExecutionPolicy(**policy)
    return ServiceConfig(**values)
