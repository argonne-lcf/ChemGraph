"""Hermetic tests for the scripted Multiwfn runner and tool wrappers."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from chemgraph.schemas.multiwfn_schema import MultiwfnInputSchema
from chemgraph.tools.multiwfn_core import run_multiwfn_core


def _write_fake_multiwfn(path: Path) -> Path:
    """Create a small executable that behaves like a menu-driven program."""
    path.write_text(
        f"""#!{sys.executable}
import os
from pathlib import Path
import sys
import time

commands = sys.stdin.read()
print(f"input={{sys.argv[1]}}")
print(f"Multiwfnpath={{os.environ.get('Multiwfnpath')}}")
print(commands, end="")
print("fake stderr", file=sys.stderr)
Path("artifact.txt").write_text(commands, encoding="utf-8")

if "sleep" in commands:
    time.sleep(60)
if "fail" in commands:
    raise SystemExit(7)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def configured_multiwfn(monkeypatch, tmp_path):
    executable = _write_fake_multiwfn(tmp_path / "Multiwfn")
    home = tmp_path / "multiwfn_home"
    home.mkdir()
    input_file = tmp_path / "sample.fchk"
    input_file.write_text("wavefunction", encoding="utf-8")
    log_dir = tmp_path / "logs"

    monkeypatch.setenv("MULTIWFN_EXE", str(executable))
    monkeypatch.setenv("MULTIWFN_HOME", str(home))
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(log_dir))
    return executable, home, input_file, log_dir


def test_run_multiwfn_core_captures_streams_and_artifacts(
    configured_multiwfn, capsys
):
    executable, home, input_file, log_dir = configured_multiwfn
    params = MultiwfnInputSchema(
        input_file=str(input_file),
        menu_inputs=["1", "", "-10"],
        timeout_s=5,
    )

    result = run_multiwfn_core(params)

    assert result["status"] == "success"
    assert result["return_code"] == 0
    assert result["executable"] == str(executable.resolve())
    assert result["input_file"] == str(input_file.resolve())
    assert Path(result["run_directory"]).parent == log_dir.resolve()
    assert Path(result["stdin_file"]).read_text(encoding="utf-8") == "1\n\n-10\n"
    assert f"Multiwfnpath={home.resolve()}" in result["stdout_tail"]
    assert "1\n\n-10\n" in result["stdout_tail"]
    assert "fake stderr" in result["stderr_tail"]
    assert result["artifacts"] == [
        str(Path(result["run_directory"]) / "artifact.txt")
    ]
    assert capsys.readouterr() == ("", "")


def test_run_multiwfn_core_creates_unique_run_directories(configured_multiwfn):
    _, _, input_file, _ = configured_multiwfn
    params = MultiwfnInputSchema(input_file=str(input_file), menu_inputs=["-10"])

    first = run_multiwfn_core(params)
    second = run_multiwfn_core(params)

    assert first["run_directory"] != second["run_directory"]


def test_run_multiwfn_core_reports_nonzero_exit(configured_multiwfn):
    _, _, input_file, _ = configured_multiwfn

    result = run_multiwfn_core(
        MultiwfnInputSchema(input_file=str(input_file), menu_inputs=["fail"])
    )

    assert result["status"] == "failure"
    assert result["return_code"] == 7
    assert result["artifacts"]


def test_run_multiwfn_core_times_out(configured_multiwfn):
    _, _, input_file, _ = configured_multiwfn

    result = run_multiwfn_core(
        MultiwfnInputSchema(
            input_file=str(input_file),
            menu_inputs=["sleep"],
            timeout_s=0.1,
        )
    )

    assert result["status"] == "timeout"
    assert result["return_code"] is not None
    assert result["duration_s"] < 5


def test_run_multiwfn_core_resolves_input_from_log_dir(configured_multiwfn):
    _, _, input_file, log_dir = configured_multiwfn
    log_dir.mkdir()
    log_input = log_dir / input_file.name
    log_input.write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_multiwfn_core(
        MultiwfnInputSchema(input_file=input_file.name, menu_inputs=["-10"])
    )

    assert result["input_file"] == str(log_input.resolve())


def test_run_multiwfn_core_requires_configured_executable(monkeypatch, tmp_path):
    monkeypatch.delenv("MULTIWFN_EXE", raising=False)
    input_file = tmp_path / "sample.fchk"
    input_file.write_text("wavefunction", encoding="utf-8")

    with pytest.raises(ValueError, match="MULTIWFN_EXE is not set"):
        run_multiwfn_core(
            MultiwfnInputSchema(input_file=str(input_file), menu_inputs=["-10"])
        )


def test_run_multiwfn_core_rejects_missing_executable(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTIWFN_EXE", str(tmp_path / "missing-Multiwfn"))
    input_file = tmp_path / "sample.fchk"
    input_file.write_text("wavefunction", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_multiwfn_core(
            MultiwfnInputSchema(input_file=str(input_file), menu_inputs=["-10"])
        )


def test_run_multiwfn_core_rejects_non_executable(monkeypatch, tmp_path):
    executable = tmp_path / "Multiwfn"
    executable.write_text("not executable", encoding="utf-8")
    executable.chmod(0o644)
    monkeypatch.setenv("MULTIWFN_EXE", str(executable))
    input_file = tmp_path / "sample.fchk"
    input_file.write_text("wavefunction", encoding="utf-8")

    with pytest.raises(PermissionError, match="is not executable"):
        run_multiwfn_core(
            MultiwfnInputSchema(input_file=str(input_file), menu_inputs=["-10"])
        )


def test_run_multiwfn_core_rejects_missing_input(configured_multiwfn):
    with pytest.raises(FileNotFoundError, match="input file not found"):
        run_multiwfn_core(
            MultiwfnInputSchema(input_file="missing.fchk", menu_inputs=["-10"])
        )


@pytest.mark.parametrize("value", ["one\ntwo", "one\rtwo", "one\x00two"])
def test_multiwfn_schema_rejects_non_single_line_responses(value):
    with pytest.raises(ValidationError):
        MultiwfnInputSchema(input_file="sample.fchk", menu_inputs=[value])


def test_single_agent_binds_run_multiwfn():
    from chemgraph.graphs.single_agent import construct_single_agent_graph
    from chemgraph.tools.multiwfn_tools import run_multiwfn

    graph = construct_single_agent_graph(MagicMock(), tools=[run_multiwfn])

    assert graph is not None


@pytest.mark.asyncio
async def test_dedicated_multiwfn_mcp_server(configured_multiwfn):
    from fastmcp import Client

    from chemgraph.mcp.multiwfn_mcp import mcp

    _, _, input_file, _ = configured_multiwfn

    async with Client(mcp) as client:
        listed = await client.list_tools()
        result = await client.call_tool(
            "run_multiwfn",
            {
                "params": {
                    "input_file": str(input_file),
                    "menu_inputs": ["-10"],
                    "timeout_s": 5,
                }
            },
        )

    assert [tool.name for tool in listed] == ["run_multiwfn"]
    assert result.structured_content["status"] == "success"
