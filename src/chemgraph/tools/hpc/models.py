"""Non-secret configuration and batch requests for scheduler-facing tools."""

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in value
        or "\\" in value
    ):
        raise ValueError("Expected a relative POSIX path without traversal.")
    if not path.parts or path.parts[0].startswith(".hpc"):
        raise ValueError("Reserved or empty path.")
    return str(path)


class Resources(StrictModel):
    node_count: int = Field(default=1, ge=1)
    duration: int = Field(default=300, ge=1, description="Walltime in seconds")
    filesystems: str = "home:eagle"


class HPCTarget(StrictModel):
    compute_resource: str = Field(min_length=1)
    storage_resource: str = Field(min_length=1)
    local_collection: str = Field(min_length=1)
    remote_collection: str = Field(min_length=1)
    local_root: str
    local_collection_root: str
    remote_collection_root: str
    remote_root: str
    project: str = Field(min_length=1)
    queue: str = Field(min_length=1)
    resources: Resources = Field(default_factory=Resources)
    setup_script: str | None = None
    python_executable: str | None = None

    @field_validator(
        "local_root", "local_collection_root", "remote_collection_root", "remote_root"
    )
    @classmethod
    def absolute_root(cls, value):
        if (
            not value.startswith("/")
            or ".." in PurePosixPath(value).parts
            or "\x00" in value
        ):
            raise ValueError("Roots must be absolute paths without traversal.")
        return str(PurePosixPath(value))

    def local_path(self, path: Path) -> str:
        return str(
            PurePosixPath(self.local_collection_root)
            / path.relative_to(Path(self.local_root).resolve()).as_posix()
        )

    def collection_path(self, compute_path: str) -> str:
        return str(
            PurePosixPath(self.remote_collection_root)
            / PurePosixPath(compute_path).relative_to(self.remote_root)
        )


class HPCConfig(StrictModel):
    targets: dict[str, HPCTarget] = Field(default_factory=dict)


class BatchRequest(StrictModel):
    target: str
    launch_script: str
    arguments: list[str] = Field(default_factory=list)
    resources: Resources | None = None
    project: str | None = None
    queue: str | None = None
    stdout: str = "stdout.txt"
    stderr: str = "stderr.txt"
    expected_outputs: list[str] = Field(default_factory=list)

    @field_validator("launch_script", "stdout", "stderr")
    @classmethod
    def check_path(cls, value):
        return relative_path(value)

    @field_validator("expected_outputs")
    @classmethod
    def check_outputs(cls, values):
        return [relative_path(value) for value in values]
