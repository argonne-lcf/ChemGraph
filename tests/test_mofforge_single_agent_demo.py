"""Tests for the thin ChemGraph + four MCP servers example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "mofforge_example"
    / "demo_single_agent_all_mcp.py"
)
SPEC = importlib.util.spec_from_file_location("mofforge_agent_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def _args():
    return SimpleNamespace(
        backend="local",
        compute_system="local",
        mofforge_python=sys.executable,
        fairchem_python=sys.executable,
        pacmof2_python=sys.executable,
        graspa_python=sys.executable,
    )


def test_server_configs_start_the_expected_mcp_modules():
    configs = demo.build_server_configs(_args())

    assert set(configs) == set(demo.SERVER_MODULES)
    for name, module in demo.SERVER_MODULES.items():
        assert configs[name]["command"] == str(Path(sys.executable).resolve())
        assert configs[name]["args"] == [
            "-u",
            "-m",
            module,
            "--transport",
            "stdio",
        ]


def test_only_generic_hpc_tools_are_prefixed():
    assert demo._prefix_tools("mofforge") is False
    assert demo._prefix_tools("fairchem") is True
    assert demo._prefix_tools("pacmof2") is True
    assert demo._prefix_tools("graspa") is True


def test_server_environment_excludes_llm_credentials(monkeypatch):
    monkeypatch.setenv("MOFFORGE_LOG_DIR", "/tmp/mofforge")
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-mcp-servers")

    env = demo._server_environment("local", "local")

    assert env["MOFFORGE_LOG_DIR"] == "/tmp/mofforge"
    assert env["CHEMGRAPH_EXECUTION_BACKEND"] == "local"
    assert "OPENAI_API_KEY" not in env


@pytest.mark.asyncio
async def test_run_chemgraph_uses_standard_single_agent(monkeypatch):
    calls = {}

    class FakeChemGraph:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        async def run(self, query):
            calls["query"] = query
            return "standard-agent-result"

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.ChemGraph",
        FakeChemGraph,
    )
    tools = [SimpleNamespace(name="mofforge_validate")]

    result = await demo.run_chemgraph(
        tools,
        model="test-model",
        query="test query",
        recursion_limit=12,
    )

    assert result == "standard-agent-result"
    assert calls["query"] == "test query"
    assert calls["kwargs"]["workflow_type"] == "single_agent"
    assert calls["kwargs"]["tools"] is tools
    assert calls["kwargs"]["return_option"] == "last_message"
    assert "system_prompt" not in calls["kwargs"]
