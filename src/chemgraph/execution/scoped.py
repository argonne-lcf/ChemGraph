"""Caller-scoped, durable native-tool execution for embedding applications.

This is an application boundary, not an operating-system sandbox. Runtime
executables and model installations remain operator-controlled resources.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

EXECUTION_CONTRACT_VERSION = 1
ARTIFACT_KEY = "chemgraph_artifacts"


class ExecutionController(Protocol):
    async def execute_tool(
        self, context, call: dict, invoke: Callable[[], Awaitable[dict]]
    ) -> dict:
        """Durably accept before invoking; never replay uncertain side effects."""
        ...


@dataclass
class ExecutionContext:
    owner: str
    thread_id: str
    task_id: str
    workspace: Path
    controller: ExecutionController
    # The embedding runtime retains this mapping for the lifetime of a thread.
    state: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    model_roots: tuple[Path, ...] = ()

    def __post_init__(self):
        if not all((self.owner, self.thread_id, self.task_id)):
            raise ValueError("Execution requires caller, thread and task identities")
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model_roots = tuple(Path(root).resolve() for root in self.model_roots)


_context: ContextVar[ExecutionContext | None] = ContextVar(
    "chemgraph_execution", default=None
)


def current_execution() -> ExecutionContext | None:
    return _context.get()


@contextmanager
def execution_context(context: ExecutionContext):
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def scoped_path(path) -> str | None:
    """Resolve a tool-controlled path, rejecting traversal and symlink escapes."""
    context = current_execution()
    if context is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = context.workspace / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(context.workspace):
        raise PermissionError("Tool path is outside this conversation's workspace")
    if ".chemgraph-artifacts" in resolved.relative_to(context.workspace).parts:
        raise PermissionError("Artifact snapshots are read-only transport records")
    return str(resolved)


def scoped_calculator(calculator: dict) -> dict:
    """Keep calculator programs operator-controlled and their outputs scoped."""
    context = current_execution()
    if context is None:
        return calculator
    calculator = dict(calculator)
    if calculator.get("command") or calculator.get("profile"):
        raise PermissionError(
            "Calculator commands/profiles must be configured by the operator"
        )
    if "directory" in calculator:
        calculator["directory"] = scoped_path(calculator["directory"])
    model = calculator.get("model")
    if model is not None:
        value = str(model)
        path = Path(value).expanduser()
        if "://" in value:
            raise PermissionError(
                "Scoped calculators require installed model resources"
            )
        if (
            isinstance(model, Path)
            or "/" in value
            or "\\" in value
            or path.suffix in {".model", ".pt", ".pth", ".ckpt", ".jit"}
            or path.exists()
        ):
            path = path.resolve()
            if not any(path.is_relative_to(root) for root in context.model_roots):
                raise PermissionError(
                    "Model path is outside operator-configured model roots"
                )
            calculator["model"] = str(path)
    return calculator


def standalone_log_record(record):
    """A process-wide file handler must never collect another caller's logs."""
    return current_execution() is None


def _files(workspace):
    """Capture ordinary output files, excluding transport snapshots and logs."""
    files = {}
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if (
            any(part.startswith(".") for part in relative.parts)
            or path.suffix == ".log"
        ):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if not path.resolve().is_relative_to(workspace):
            continue
        content = path.read_bytes()
        files[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    return files


def _capture(context, call_id, before):
    """Snapshot the bytes written by this tool before another call can replace them."""
    artifacts = []
    for relative, digest in _files(context.workspace).items():
        if before.get(relative) == digest:
            continue
        source = context.workspace / relative
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        key = hashlib.sha256(
            f"{context.task_id}:{call_id}:{relative}:{digest}".encode()
        ).hexdigest()
        destination = context.workspace / ".chemgraph-artifacts" / key / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        artifacts.append(
            {
                "path": str(destination),
                "media_type": mimetypes.guess_type(source.name)[0]
                or "application/octet-stream",
                "task_id": context.task_id,
                "tool_call_id": call_id,
                "source_path": relative,
                "sha256": digest,
            }
        )
    return artifacts


def encode_result(value) -> dict:
    kind, payload = JsonPlusSerializer().dumps_typed(value)
    return {"type": kind, "data": base64.b64encode(payload).decode("ascii")}


def decode_result(value):
    return JsonPlusSerializer().loads_typed(
        (value["type"], base64.b64decode(value["data"]))
    )


def result_artifacts(state, task_id):
    """Read descriptors from live or serialized checkpoint messages."""
    artifacts = {}
    for message in state.get("messages", []):
        extra = (
            message.get("additional_kwargs", {})
            if isinstance(message, dict)
            else getattr(message, "additional_kwargs", {})
        )
        for artifact in extra.get(ARTIFACT_KEY, []):
            if artifact.get("task_id") == task_id:
                artifacts[artifact["path"]] = artifact
    return list(artifacts.values())


async def execute_scoped_tool(spec, request, handler):
    context = current_execution()
    if context is None or spec.interactive:
        return await handler(request)
    if spec.executes_code:
        raise PermissionError(
            "Host code execution is unavailable in scoped tool catalogs"
        )
    call = request.tool_call
    async with context.lock:

        async def invoke():
            before = await asyncio.to_thread(_files, context.workspace)
            result = await handler(request)
            artifacts = await asyncio.to_thread(_capture, context, call["id"], before)
            if isinstance(result, ToolMessage):
                result = result.model_copy(
                    update={
                        "additional_kwargs": {
                            **result.additional_kwargs,
                            ARTIFACT_KEY: artifacts,
                        },
                    }
                )
            elif artifacts:
                raise TypeError(
                    "Artifact-producing native tools must return a ToolMessage"
                )
            return {"result": encode_result(result), "artifacts": artifacts}

        # Controller errors deliberately propagate; callbacks must not authorize work.
        saved = await context.controller.execute_tool(context, call, invoke)
        return decode_result(saved["result"])
