"""Hermetic tests for the four-server mofforge single-agent demo."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "mofforge_example"
    / "demo_single_agent_all_mcp.py"
)
SPEC = importlib.util.spec_from_file_location(
    "demo_single_agent_all_mcp",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def _args(**overrides):
    values = {
        "backend": "local",
        "compute_system": "local",
        "mofforge_python": sys.executable,
        "fairchem_python": sys.executable,
        "pacmof2_python": sys.executable,
        "graspa_python": sys.executable,
        "fairchem_ppn": 2,
        "fairchem_ngpus_per_process": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _tool(name: str):
    return SimpleNamespace(name=name)


def test_build_server_configs_uses_expected_modules_and_resource_flags():
    configs = demo.build_server_configs(_args())

    assert set(configs) == {"mofforge", "fairchem", "pacmof2", "graspa"}
    for server_name, module in demo.SERVER_MODULES.items():
        config = configs[server_name]
        assert config["transport"] == "stdio"
        assert config["command"] == str(Path(sys.executable).resolve())
        assert config["args"][:3] == ["-u", "-m", module]
        assert config["args"][-2:] == ["--transport", "stdio"]

    fairchem_args = configs["fairchem"]["args"]
    assert fairchem_args[fairchem_args.index("--ppn") + 1] == "2"
    assert (
        fairchem_args[fairchem_args.index("--ngpus-per-process") + 1]
        == "1"
    )


def test_forwarded_environment_keeps_runtime_config_but_not_llm_keys(
    monkeypatch,
):
    monkeypatch.setenv("MOFFORGE_LOG_DIR", "/tmp/mofforge-output")
    monkeypatch.setenv("GLOBUS_COMPUTE_ENDPOINT_ID", "endpoint-id")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-tool-server")

    env = demo._forwarded_environment("globus_compute", "polaris")

    assert env["MOFFORGE_LOG_DIR"] == "/tmp/mofforge-output"
    assert env["GLOBUS_COMPUTE_ENDPOINT_ID"] == "endpoint-id"
    assert env["CHEMGRAPH_EXECUTION_BACKEND"] == "globus_compute"
    assert env["COMPUTE_SYSTEM"] == "polaris"
    assert "OPENAI_API_KEY" not in env


def test_only_mofforge_tools_remain_unprefixed():
    assert demo._tool_prefix_enabled("mofforge") is False
    assert demo._tool_prefix_enabled("fairchem") is True
    assert demo._tool_prefix_enabled("pacmof2") is True
    assert demo._tool_prefix_enabled("graspa") is True


def test_validate_tool_inventory_accepts_required_names():
    grouped = {
        server_name: [_tool(name) for name in sorted(required)]
        for server_name, required in demo.REQUIRED_TOOLS.items()
    }

    demo.validate_tool_inventory(grouped)


def test_validate_tool_inventory_reports_missing_tool():
    grouped = {
        server_name: [_tool(name) for name in sorted(required)]
        for server_name, required in demo.REQUIRED_TOOLS.items()
    }
    grouped["graspa"] = []

    with pytest.raises(RuntimeError, match="graspa.*run_graspa_ensemble"):
        demo.validate_tool_inventory(grouped)


def test_validate_tool_inventory_rejects_duplicate_names():
    grouped = {
        server_name: [_tool(name) for name in sorted(required)]
        for server_name, required in demo.REQUIRED_TOOLS.items()
    }
    grouped["mofforge"].append(_tool("shared_name"))
    grouped["fairchem"].append(_tool("shared_name"))

    with pytest.raises(RuntimeError, match="Duplicate MCP tool names"):
        demo.validate_tool_inventory(grouped)


def test_resolve_python_honors_server_environment(monkeypatch):
    monkeypatch.setenv("DEMO_TEST_PYTHON", sys.executable)
    resolved = demo._resolve_python(None, "DEMO_TEST_PYTHON")
    assert resolved == str(Path(sys.executable).resolve())


def test_parser_defaults_to_async_inventory_capable_cli(monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_EXECUTION_BACKEND", raising=False)
    parser = demo.build_parser()
    args = parser.parse_args(["--list-tools-only"])

    assert args.list_tools_only is True
    assert args.backend == "local"
    assert args.query == demo.DEFAULT_QUERY
    assert args.model
