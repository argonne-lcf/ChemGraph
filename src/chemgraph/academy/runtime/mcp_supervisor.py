"""Spawn per-rank MCP server subprocesses, wait for readiness, connect."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

from chemgraph.academy.core.campaign import MCPServerSpec

logger = logging.getLogger(__name__)

_READINESS_TIMEOUT_S = 30.0
_READINESS_POLL_INTERVAL_S = 0.25
_SHUTDOWN_TIMEOUT_S = 5.0


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MCPServerSupervisor:
    """Per-rank MCP subprocess lifecycle and client wiring."""

    def __init__(self, specs: list[MCPServerSpec], run_dir: Path) -> None:
        self._specs = list(specs)
        self._run_dir = Path(run_dir)
        self._log_dir = self._run_dir / "mcp_logs"
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._log_handles: dict[str, object] = {}
        self._urls: dict[str, str] = {}

    @property
    def urls(self) -> dict[str, str]:
        return dict(self._urls)

    async def start_all(self) -> dict[str, str]:
        if not self._specs:
            return {}
        self._log_dir.mkdir(parents=True, exist_ok=True)
        for spec in self._specs:
            port = _pick_free_port()
            url = f"http://127.0.0.1:{port}/mcp/"
            cmd = shlex.split(spec.command) + [
                "--transport",
                "streamable_http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            env = {**os.environ, **spec.env}
            log_path = self._log_dir / f"{spec.name}.log"
            log_handle = log_path.open("ab")
            logger.info(
                "spawning MCP server %s on port %d: %s",
                spec.name,
                port,
                " ".join(cmd),
            )
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            self._processes[spec.name] = proc
            self._log_handles[spec.name] = log_handle
            self._urls[spec.name] = url
        await self._await_all_ready()
        return dict(self._urls)

    async def get_tools(
        self,
        server_names: tuple[str, ...] | None = None,
    ) -> list[BaseTool]:
        if not self._urls:
            return []
        wanted = tuple(server_names) if server_names else tuple(self._urls)
        unknown = sorted(set(wanted) - set(self._urls))
        if unknown:
            raise RuntimeError(
                f"agent requested unknown MCP servers: {unknown}; "
                f"available: {sorted(self._urls)}",
            )
        connections = {
            name: self._urls[name]
            for name in wanted
        }
        tools: list[BaseTool] = []
        tool_names: set[str] = set()
        for server_name, url in connections.items():
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    for mcp_tool in listed.tools:
                        if mcp_tool.name in tool_names:
                            raise RuntimeError(
                                f"duplicate MCP tool name {mcp_tool.name!r} "
                                f"from server {server_name!r}",
                            )
                        tool_names.add(mcp_tool.name)
                        tools.append(
                            _langchain_tool(
                                server_name=server_name,
                                server_url=url,
                                tool_name=mcp_tool.name,
                                description=mcp_tool.description
                                or f"MCP tool {mcp_tool.name}.",
                                input_schema=mcp_tool.inputSchema,
                            ),
                        )
        return tools

    async def shutdown(self) -> None:
        for name, proc in list(self._processes.items()):
            if proc.poll() is not None:
                continue
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()

        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_S
        for name, proc in list(self._processes.items()):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("MCP server %s did not exit; killing", name)
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2)

        for handle in self._log_handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        self._processes.clear()
        self._log_handles.clear()
        self._urls.clear()

    async def _await_all_ready(self) -> None:
        deadline = time.monotonic() + _READINESS_TIMEOUT_S
        pending = dict(self._urls)
        async with httpx.AsyncClient(timeout=2.0) as client:
            while pending and time.monotonic() < deadline:
                ready_now: list[str] = []
                for name, url in pending.items():
                    proc = self._processes[name]
                    if proc.poll() is not None:
                        log_tail = self._tail_log(name)
                        raise RuntimeError(
                            f"MCP server {name!r} exited before readiness "
                            f"(returncode={proc.returncode}). Last log lines:\n"
                            f"{log_tail}",
                        )
                    try:
                        response = await client.get(url)
                        if response.status_code < 500:
                            ready_now.append(name)
                    except httpx.RequestError:
                        pass
                for name in ready_now:
                    logger.info("MCP server %s ready at %s", name, pending[name])
                    pending.pop(name)
                if pending:
                    await asyncio.sleep(_READINESS_POLL_INTERVAL_S)
            if pending:
                stuck = sorted(pending)
                tails = "\n".join(
                    f"=== {name} ===\n{self._tail_log(name)}"
                    for name in stuck
                )
                raise RuntimeError(
                    f"MCP servers did not become ready within "
                    f"{_READINESS_TIMEOUT_S:.0f}s: {stuck}\n{tails}",
                )

    def _tail_log(self, name: str, n: int = 40) -> str:
        path = self._log_dir / f"{name}.log"
        if not path.exists():
            return "(no log file)"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(log unreadable)"
        return "\n".join(text.splitlines()[-n:])


def _langchain_tool(
    *,
    server_name: str,
    server_url: str,
    tool_name: str,
    description: str,
    input_schema: dict[str, Any],
) -> BaseTool:
    async def call_mcp_tool(**kwargs: Any) -> Any:
        return await _call_mcp_tool(
            server_url=server_url,
            tool_name=tool_name,
            arguments=kwargs,
        )

    call_mcp_tool.__name__ = f"{server_name}_{tool_name}"
    return StructuredTool.from_function(
        coroutine=call_mcp_tool,
        name=tool_name,
        description=description,
        args_schema=input_schema,
        metadata={
            "chemgraph_academy_tool_kind": "science_tool",
            "mcp_server": server_name,
        },
    )


async def _call_mcp_tool(
    *,
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    async with streamablehttp_client(server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _serialize_call_tool_result(result)


def _serialize_call_tool_result(result: CallToolResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "is_error": bool(result.isError),
        "content": [
            _json_safe(block)
            for block in result.content
        ],
    }
    if result.structuredContent is not None:
        payload["structured_content"] = _json_safe(result.structuredContent)
    if result.isError:
        payload["status"] = "error"
    else:
        payload["status"] = "ok"
    return payload


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
