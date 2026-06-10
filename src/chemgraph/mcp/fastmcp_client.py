"""In-process client adapter for FastMCP tool modules."""

from __future__ import annotations

import importlib
import json
import uuid
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Protocol

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ToolInvocation(BaseModel):
    """A normalized record of one agent-requested FastMCP tool call."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    agent_id: str | None = None
    role: str | None = None
    correlation_id: str = Field(default_factory=lambda: f"call-{uuid.uuid4()}")


class ToolResult(BaseModel):
    """Normalized result from a campaign FastMCP tool call."""

    model_config = ConfigDict(extra="allow")

    tool_name: str
    status: str
    result: Any = None
    error: str | None = None
    correlation_id: str


class FastMCPToolSpec(Protocol):
    """Structural interface for config-declared FastMCP tools."""

    name: str
    module: str
    tool: str
    description: str | None


class FastMCPExecutionSpec(Protocol):
    """Structural interface for backend configuration used by CGFastMCP."""

    backend: str | None
    system: str | None
    config_path: str | None
    options: Mapping[str, Any]


def load_fastmcp_tool_module(
    module_name: str,
    *,
    cache: dict[str, Any] | None = None,
) -> Any:
    """Return a module's top-level FastMCP object declared by a campaign tool."""
    if cache is not None and module_name in cache:
        return cache[module_name]

    module = importlib.import_module(module_name)
    try:
        server = module.mcp
    except AttributeError as exc:
        raise RuntimeError(
            f"FastMCP tool module {module_name!r} does not expose "
            "a top-level 'mcp' object",
        ) from exc

    if cache is not None:
        cache[module_name] = server
    return server


async def fastmcp_tool_schemas(
    specs: Sequence[FastMCPToolSpec],
) -> list[dict[str, Any]]:
    """Build OpenAI tool schemas from declared FastMCP ToolSpecs."""
    schemas: list[dict[str, Any]] = []
    module_cache: dict[str, Any] = {}
    tools_cache: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.module not in tools_cache:
            tools = await load_fastmcp_tool_module(
                spec.module,
                cache=module_cache,
            ).list_tools()
            tools_cache[spec.module] = {
                _fastmcp_tool_name(tool): _fastmcp_tool_payload(tool)
                for tool in tools
            }
        try:
            tool_payload = tools_cache[spec.module][spec.tool]
        except KeyError as exc:
            raise RuntimeError(
                f"FastMCP module {spec.module!r} does not expose tool "
                f"{spec.tool!r}",
            ) from exc
        schemas.append(_openai_tool_schema(spec, tool_payload))
    return schemas


def _fastmcp_tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", ""))


def _fastmcp_tool_payload(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return dict(tool)
    if hasattr(tool, "model_dump"):
        return tool.model_dump(mode="json")
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "inputSchema": getattr(tool, "inputSchema", None),
    }


def _openai_tool_schema(
    spec: FastMCPToolSpec,
    tool_payload: dict[str, Any],
) -> dict[str, Any]:
    parameters = _sanitize_input_schema(
        tool_payload.get("inputSchema") or {"type": "object", "properties": {}},
    )
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description
            or str(tool_payload.get("description") or ""),
            "parameters": parameters,
        },
    }


def _sanitize_input_schema(schema: Any) -> dict[str, Any]:
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(mode="json")
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": False}
    sanitized = json.loads(json.dumps(schema))
    sanitized.setdefault("type", "object")
    sanitized.setdefault("properties", {})
    sanitized.setdefault("additionalProperties", False)
    return sanitized


def serialize_fastmcp_result(result: Any) -> Any:
    """Convert FastMCP content blocks to JSON-friendly values."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        values = [serialize_fastmcp_result(item) for item in result]
        structured = _first_structured_result(values)
        if structured is not None:
            return structured
        json_text = _first_json_text_result(values)
        if json_text is not None:
            return json_text
        return values
    return str(result)


def _first_structured_result(values: list[Any]) -> dict[str, Any] | None:
    for value in values:
        if isinstance(value, dict) and (
            "results" in value
            or "batch_id" in value
            or "progress_pct" in value
            or value.get("status") in {"completed", "submitted"}
        ):
            return value
        if isinstance(value, list):
            nested = _first_structured_result(value)
            if nested is not None:
                return nested
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            try:
                parsed = json.loads(value["text"])
            except json.JSONDecodeError:
                continue
            nested = _first_structured_result([parsed])
            if nested is not None:
                return nested
    return None


def _first_json_text_result(values: list[Any]) -> Any | None:
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            try:
                return json.loads(value["text"])
            except json.JSONDecodeError:
                continue
        if isinstance(value, list):
            nested = _first_json_text_result(value)
            if nested is not None:
                return nested
    return None


class FastMCPToolInvoker:
    """Invoke allowed tools through in-process FastMCP modules."""

    def __init__(
        self,
        *,
        specs: Sequence[FastMCPToolSpec],
        execution: FastMCPExecutionSpec,
        run_dir: str | Path,
    ) -> None:
        self.specs = {spec.name: spec for spec in specs}
        self.execution = execution
        self.run_dir = Path(run_dir)
        self._module_cache: dict[str, Any] = {}
        self._available_cache: dict[str, set[str]] = {}

    def names(self) -> list[str]:
        return sorted(self.specs)

    async def verify_allowed_tools(self) -> list[str]:
        """Return tools missing from their declared FastMCP module."""
        missing: list[str] = []
        for spec in self.specs.values():
            try:
                available = await self._available_tool_names(spec.module)
            except Exception:  # noqa: BLE001 - caller needs aggregate missing names
                missing.append(spec.name)
                continue
            if spec.tool not in available:
                missing.append(spec.name)
        return missing

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        spec = self.specs.get(invocation.tool_name)
        if spec is None:
            raise KeyError(
                f"unknown campaign FastMCP tool: {invocation.tool_name}",
            )

        try:
            available = await self._available_tool_names(spec.module)
            if spec.tool not in available:
                raise KeyError(
                    f"FastMCP module {spec.module!r} does not expose "
                    f"tool {spec.tool!r}",
                )
            mcp = self._fastmcp_module(spec.module)
            _configure_fastmcp_backend(
                mcp,
                module_name=spec.module,
                execution=self.execution,
                run_dir=self.run_dir,
            )
            from chemgraph.observability.events import WorkflowEventContext
            from chemgraph.observability.events import workflow_event_context

            context = WorkflowEventContext(
                run_id=self.run_dir.name,
                run_dir=str(self.run_dir),
                agent_id=invocation.agent_id,
                role=invocation.role,
                parent_span_id=invocation.correlation_id,
                tool_name=invocation.tool_name,
            )
            with workflow_event_context(
                jsonl_path=self.run_dir / "events.jsonl",
                context=context,
            ):
                result = await mcp.call_tool(spec.tool, invocation.arguments)
        except Exception as exc:  # noqa: BLE001 - preserve tool failure as data
            return ToolResult(
                tool_name=invocation.tool_name,
                status="error",
                error=repr(exc),
                correlation_id=invocation.correlation_id,
            )

        return ToolResult(
            tool_name=invocation.tool_name,
            status="success",
            result=serialize_fastmcp_result(result),
            correlation_id=invocation.correlation_id,
        )

    async def _available_tool_names(self, module_name: str) -> set[str]:
        if module_name not in self._available_cache:
            tools = await self._fastmcp_module(module_name).list_tools()
            self._available_cache[module_name] = {
                str(getattr(tool, "name", ""))
                if not isinstance(tool, dict)
                else str(tool.get("name", ""))
                for tool in tools
            }
        return self._available_cache[module_name]

    def _fastmcp_module(self, module_name: str) -> Any:
        return load_fastmcp_tool_module(module_name, cache=self._module_cache)


def _configure_fastmcp_backend(
    mcp: Any,
    *,
    module_name: str,
    execution: FastMCPExecutionSpec,
    run_dir: str | Path,
) -> None:
    """Configure a CGFastMCP backend without initialising compute resources."""
    if not hasattr(mcp, "init_backend"):
        return
    if getattr(mcp, "_backend_kwargs", None) is not None:
        return

    kwargs: dict[str, Any] = dict(execution.options)
    if execution.config_path:
        kwargs["config_path"] = execution.config_path
    if execution.backend:
        kwargs["backend_name"] = execution.backend
    if execution.system:
        kwargs["system"] = execution.system

    tracker_name = module_name.replace(".", "_")
    tracker_path = Path(run_dir) / f"{tracker_name}_jobs.json"
    mcp.init_backend(
        tracker_kwargs={"persist_file": str(tracker_path)},
        **kwargs,
    )


async def build_fastmcp_tool_invoker(
    *,
    specs: Sequence[FastMCPToolSpec],
    execution: FastMCPExecutionSpec,
    run_dir: str | Path,
    agent_name: str,
) -> FastMCPToolInvoker:
    """Build and verify one agent's in-process FastMCP tool invoker."""
    invoker = FastMCPToolInvoker(
        specs=list(specs),
        execution=execution,
        run_dir=run_dir,
    )
    missing = await invoker.verify_allowed_tools()
    if missing:
        raise RuntimeError(
            f"Could not resolve requested FastMCP tools for {agent_name}: {missing}",
        )
    return invoker
