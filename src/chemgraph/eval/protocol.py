"""Strict, dataset-free protocol shared by the evaluation controller and worker."""

import json
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL_VERSION = 1
RESULT_FILE = ".chemgraph-result.json"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ExecutionSettings(StrictModel):
    """Execution options only; no dataset, checkpoint, or host paths."""

    recursion_limit: int = Field(default=200, ge=1)
    structured_output: bool = True
    deepagent_auto_approve: Literal[True] = True

    @classmethod
    def from_config(cls, config):
        return cls(**{key: getattr(config, key) for key in cls.model_fields})


def hosted_argo_url(value: str) -> str:
    """Accept only TLS Argo API roots, without credentials or URL parameters."""
    url = urlsplit(value)
    if (url.scheme != "https" or url.hostname != "apps.inside.anl.gov"
            or url.port not in (None, 443) or url.username or url.password
            or url.query or url.fragment or url.path.rstrip("/") not in
            {"/argoapi", "/argoapi/v1", "/argotestapi", "/argotestapi/v1"}):
        raise ValueError("Podman evaluation requires a hosted HTTPS Argo API endpoint")
    return value


class WorkerRequest(StrictModel):
    version: Literal[1] = PROTOCOL_VERSION
    query: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    model: str = Field(pattern=r"^argo:[a-zA-Z0-9_.-]+$")
    base_url: str
    argo_user: str = Field(min_length=1)
    execution: ExecutionSettings

    _hosted_url = field_validator("base_url")(hosted_argo_url)


class WorkerResult(StrictModel):
    query_id: str
    query: str
    workspace: Literal["/workspace"]
    tool_calls: list[dict[str, Any]]
    result: Any
    status: Literal["completed", "initialization_error", "execution_error", "formatting_error"]
    error: str | None = None
    structured_output: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    formatter_responses: list[str] | None = None
    elapsed_seconds: float = Field(ge=0)
    usage: dict[str, Any]

    @model_validator(mode="after")
    def validate_status_and_usage(self):
        if self.status != "completed" and not self.error:
            raise ValueError("Failed workers must include an error")
        if self.status == "completed" and self.error:
            raise ValueError("Completed workers cannot include an error")
        # Aggregation consumes these structures after the process is untrusted.
        from chemgraph.agent.usage import TOKEN_FIELDS

        if set(self.usage) != {"execution", "formatting", "total"}:
            raise ValueError("Invalid usage envelope")
        for key, summary in self.usage.items():
            if summary is None and key != "total":
                continue
            if not isinstance(summary, dict):
                raise ValueError("Invalid usage summary")
            required = {*TOKEN_FIELDS, "call_count", "incomplete_calls", "partial", "unreported_counts"}
            if not required.issubset(summary) or type(summary["partial"]) is not bool:
                raise ValueError("Incomplete usage summary")
            for field in (*TOKEN_FIELDS, "call_count", "incomplete_calls"):
                value = summary[field]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("Invalid usage count")
            if any(type(summary[field]) is not int for field in ("call_count", "incomplete_calls")):
                raise ValueError("Missing usage count")
            counts = summary["unreported_counts"]
            if (not isinstance(counts, dict) or set(counts) != set(TOKEN_FIELDS)
                    or any(type(v) is not int or v < 0 for v in counts.values())):
                raise ValueError("Invalid usage coverage")
        return self


class WorkerResponse(StrictModel):
    version: Literal[1]
    raw: WorkerResult


def validate_response(data: bytes, request: WorkerRequest) -> dict:
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("Worker result exceeds size limit")
    parsed = json.loads(data)
    json.dumps(parsed, allow_nan=False)  # Reject non-finite values even inside arbitrary trace JSON.
    response = WorkerResponse.model_validate(parsed)
    if (response.raw.query_id, response.raw.query) != (request.query_id, request.query):
        raise ValueError("Worker response does not match its request")
    if request.execution.structured_output and response.raw.status == "completed":
        if response.raw.structured_output is None:
            raise ValueError("Worker omitted structured output")
    return response.raw.model_dump(exclude_none=True)
