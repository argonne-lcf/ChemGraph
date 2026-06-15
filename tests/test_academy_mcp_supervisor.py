from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from chemgraph.academy.core.campaign import MCPServerSpec
from chemgraph.academy.runtime.mcp_supervisor import MCPServerSupervisor


def _pythonpath(tmp_path: Path) -> str:
    current = os.environ.get("PYTHONPATH", "")
    parts = [str(tmp_path)]
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def _write_tiny_server(tmp_path: Path) -> None:
    (tmp_path / "tiny_mcp.py").write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tiny")

@mcp.tool(name="echo", description="Echo one string.")
def echo(text: str) -> dict:
    return {"text": text}

if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=0)
""",
        encoding="utf-8",
    )


def _write_multi_tool_server(tmp_path: Path) -> None:
    """A server that advertises three tools so allowed_tools can subset it."""
    (tmp_path / "multi_mcp.py").write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("multi")

@mcp.tool(name="alpha", description="Tool alpha.")
def alpha(text: str) -> dict:
    return {"who": "alpha", "text": text}

@mcp.tool(name="beta", description="Tool beta.")
def beta(text: str) -> dict:
    return {"who": "beta", "text": text}

@mcp.tool(name="gamma", description="Tool gamma.")
def gamma(text: str) -> dict:
    return {"who": "gamma", "text": text}

if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=0)
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_mcp_supervisor_starts_server_and_gets_tools(tmp_path) -> None:
    _write_tiny_server(tmp_path)
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="tiny",
                command=f"{sys.executable} -m tiny_mcp",
                env={"PYTHONPATH": _pythonpath(tmp_path)},
            ),
        ],
        run_dir=tmp_path / "run",
    )
    try:
        urls = await supervisor.start_all()
        tools = await supervisor.get_tools(("tiny",))
        echo = next(tool for tool in tools if tool.name == "echo")
        result = await echo.ainvoke({"text": "hello"})
    finally:
        await supervisor.shutdown()

    assert sorted(urls) == ["tiny"]
    assert "echo" in {tool.name for tool in tools}
    assert result["status"] == "ok"
    assert "hello" in repr(result)


@pytest.mark.asyncio
async def test_mcp_supervisor_shutdown_terminates_process(tmp_path) -> None:
    _write_tiny_server(tmp_path)
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="tiny",
                command=f"{sys.executable} -m tiny_mcp",
                env={"PYTHONPATH": _pythonpath(tmp_path)},
            ),
        ],
        run_dir=tmp_path / "run",
    )
    await supervisor.start_all()
    proc = supervisor._processes["tiny"]

    await supervisor.shutdown()

    assert proc.poll() is not None


@pytest.mark.asyncio
async def test_mcp_supervisor_reports_server_exit_log_tail(tmp_path) -> None:
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="bad",
                command=f"{sys.executable} -c \"print('boom'); raise SystemExit(1)\"",
            ),
        ],
        run_dir=tmp_path / "run",
    )

    with pytest.raises(RuntimeError, match="boom"):
        await supervisor.start_all()

    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_mcp_supervisor_rejects_unknown_server_request(tmp_path) -> None:
    _write_tiny_server(tmp_path)
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="tiny",
                command=f"{sys.executable} -m tiny_mcp",
                env={"PYTHONPATH": _pythonpath(tmp_path)},
            ),
        ],
        run_dir=tmp_path / "run",
    )
    try:
        await supervisor.start_all()
        with pytest.raises(RuntimeError, match="available"):
            await supervisor.get_tools(("missing",))
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_get_tools_returns_all_when_no_allowed_tools(tmp_path) -> None:
    _write_multi_tool_server(tmp_path)
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="multi",
                command=f"{sys.executable} -m multi_mcp",
                env={"PYTHONPATH": _pythonpath(tmp_path)},
            ),
        ],
        run_dir=tmp_path / "run",
    )
    try:
        await supervisor.start_all()
        tools = await supervisor.get_tools(("multi",))
    finally:
        await supervisor.shutdown()

    assert {tool.name for tool in tools} == {"alpha", "beta", "gamma"}


@pytest.mark.asyncio
async def test_get_tools_filters_by_allowed_tools(tmp_path) -> None:
    _write_multi_tool_server(tmp_path)
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="multi",
                command=f"{sys.executable} -m multi_mcp",
                env={"PYTHONPATH": _pythonpath(tmp_path)},
            ),
        ],
        run_dir=tmp_path / "run",
    )
    try:
        await supervisor.start_all()
        tools = await supervisor.get_tools(
            ("multi",),
            allowed_tools=frozenset({"alpha", "gamma"}),
        )
    finally:
        await supervisor.shutdown()

    assert {tool.name for tool in tools} == {"alpha", "gamma"}


@pytest.mark.asyncio
async def test_get_tools_warns_on_whitelist_misses(tmp_path, caplog) -> None:
    _write_multi_tool_server(tmp_path)
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="multi",
                command=f"{sys.executable} -m multi_mcp",
                env={"PYTHONPATH": _pythonpath(tmp_path)},
            ),
        ],
        run_dir=tmp_path / "run",
    )
    try:
        await supervisor.start_all()
        with caplog.at_level("WARNING"):
            tools = await supervisor.get_tools(
                ("multi",),
                allowed_tools=frozenset({"alpha", "does_not_exist"}),
            )
    finally:
        await supervisor.shutdown()

    assert {tool.name for tool in tools} == {"alpha"}
    assert any(
        "does_not_exist" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_get_tools_empty_allowed_tools_returns_all(tmp_path) -> None:
    """An empty whitelist is treated as None (no filter)."""
    _write_multi_tool_server(tmp_path)
    supervisor = MCPServerSupervisor(
        [
            MCPServerSpec(
                name="multi",
                command=f"{sys.executable} -m multi_mcp",
                env={"PYTHONPATH": _pythonpath(tmp_path)},
            ),
        ],
        run_dir=tmp_path / "run",
    )
    try:
        await supervisor.start_all()
        tools = await supervisor.get_tools(
            ("multi",),
            allowed_tools=frozenset(),
        )
    finally:
        await supervisor.shutdown()

    assert {tool.name for tool in tools} == {"alpha", "beta", "gamma"}
