"""Transport-independent, immutable snapshots returned by RunService."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True)


class Interrupt(Snapshot):
    id: str
    kind: Literal["clarification", "approval", "job_wait"]
    operation_id: str | None = None
    payload: dict[str, Any]

    @property
    def response_key(self):
        return f"{self.id}:{self.kind}:{self.operation_id or ''}"


class Session(Snapshot):
    id: str
    status: Literal["ready", "active", "cancelled"] = "ready"
    current_run_id: str | None = None
    workspace: str
    configuration_id: str


class Run(Snapshot):
    id: str
    session_id: str
    status: Literal["queued", "running", "waiting_for_user", "waiting_for_job",
                    "recovery_required", "completed", "failed", "cancelled"] = "queued"
    message: str
    assistant_response: str = ""
    interrupts: tuple[Interrupt, ...] = ()
    error: str | None = None
    responses: dict[str, Any] = Field(default_factory=dict, exclude=True)


class Job(Snapshot):
    id: str
    operation_id: str
    run_id: str
    status: str
    cancel_requested: bool = False
    promoted: bool | None = None


class Artifact(Snapshot):
    id: str
    run_id: str
    source_path: str
    path: str
    media_type: str
    sha256: str
    size_bytes: int
    promoted: bool = True
