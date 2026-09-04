"""Tests for the standalone Globus ASE Deep Agent example."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "demo"
    / "demo_globus_ase_deep_agent.py"
)


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location(
        "globus_ase_deep_agent_example",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTool:
    def __init__(self, name, responses=()):
        self.name = name
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        return self.responses.pop(0)


def _tool_message(name: str, payload: dict) -> ToolMessage:
    return ToolMessage(
        name=name,
        tool_call_id=f"{name}-call",
        content=json.dumps(payload),
    )


def _tool_call_message(name: str, arguments: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": f"{name}-call",
                "type": "tool_call",
            }
        ],
    )


def test_tool_selection_is_exact_and_rejects_missing_tools(example):
    tools = [_FakeTool(name) for name in example.BOUND_TOOL_NAMES]
    tools.append(_FakeTool("run_ase"))

    selected, by_name = example.select_globus_ase_tools(tools)

    assert [tool.name for tool in selected] == list(example.BOUND_TOOL_NAMES)
    assert set(by_name) == set(example.BOUND_TOOL_NAMES)
    with pytest.raises(RuntimeError, match="get_job_results"):
        example.select_globus_ase_tools(tools[:-2])


def test_tool_selection_rejects_duplicates(example):
    tools = [_FakeTool(name) for name in example.BOUND_TOOL_NAMES]
    tools.append(_FakeTool("transfer_files"))

    with pytest.raises(RuntimeError, match="Duplicate.*transfer_files"):
        example.select_globus_ase_tools(tools)


def test_decode_tool_payload_handles_mcp_content_blocks(example):
    value = SimpleNamespace(
        content=[SimpleNamespace(text='{"status": "completed"}')]
    )

    assert example.decode_tool_payload(value) == {"status": "completed"}


def test_system_prompt_allows_autonomous_orchestration(example):
    prompt = example.GLOBUS_ASE_SYSTEM_PROMPT

    assert "orchestration agent" in prompt
    assert "autonomously" in prompt
    assert "Read offloaded tool results" in prompt
    assert "never invent simulation results" in prompt
    assert "1. Call" not in prompt
    assert "transfer_files" not in prompt
    assert "Do not use Deep Agent filesystem tools" not in prompt


def test_user_request_specifies_outcome_without_tool_sequence(example):
    request = example.build_user_request(
        input_path="/local/structures",
        is_directory=True,
        input_count=166,
        timeout=7200,
        poll_interval=15,
    )

    assert request == (
        "Stage every .xyz file in /local/structures to the configured HPC "
        "facility and run a MACE-MP small CUDA energy simulation over the "
        "staged structures. Complete the workflow and report the "
        "success/failure summary. Expect 166 input structure(s); wait up to "
        "7200 seconds and use a 15-second polling interval."
    )
    assert "transfer_files" not in request
    assert "run_ase_ensemble" not in request

    file_request = example.build_user_request(
        input_path="/local/water.xyz",
        is_directory=False,
        input_count=1,
        timeout=300,
        poll_interval=10,
    )
    assert file_request.startswith(
        "Stage the input structure at /local/water.xyz to the configured HPC"
    )


def test_discover_input_files_accepts_file_and_sorted_xyz_directory(
    example,
    tmp_path,
):
    input_dir = tmp_path / "structures"
    input_dir.mkdir()
    first = input_dir / "a.xyz"
    second = input_dir / "b.XYZ"
    ignored = input_dir / "notes.txt"
    for path in (first, second, ignored):
        path.write_text("fixture", encoding="utf-8")

    assert example.discover_input_files(first) == [first.resolve()]
    assert example.discover_input_files(input_dir) == [
        first.resolve(),
        second.resolve(),
    ]


def test_discover_input_files_rejects_missing_and_empty_directory(
    example,
    tmp_path,
):
    with pytest.raises(ValueError, match="does not exist"):
        example.discover_input_files(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no .xyz"):
        example.discover_input_files(empty)


def test_summarize_energy_results_requires_finite_successes(example, tmp_path):
    input_files = [tmp_path / "a.xyz", tmp_path / "b.xyz"]
    payload = {
        "batch_id": "batch-1",
        "status": "completed",
        "results": [
            {"index": 0, "status": "success", "potential_energy": -1.25},
            {"index": 1, "status": "success", "potential_energy": 0.75},
        ],
    }

    summary = example.summarize_energy_results(payload, "batch-1", input_files)

    assert summary["all_succeeded"] is True
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["energy_min"] == -1.25
    assert summary["energy_max"] == 0.75
    assert summary["energy_mean"] == -0.25

    payload["results"][0]["potential_energy"] = float("nan")
    summary = example.summarize_energy_results(payload, "batch-1", input_files)
    assert summary["all_succeeded"] is False
    assert summary["succeeded"] == 1
    assert summary["failures"][0] == {
        "index": 0,
        "structure": "a.xyz",
        "error_type": "InvalidEnergy",
        "message": "Invalid potential_energy: nan",
    }


def test_summarize_energy_results_reports_failures_and_missing_results(
    example,
    tmp_path,
):
    input_files = [tmp_path / "a.xyz", tmp_path / "b.xyz"]
    payload = {
        "batch_id": "batch-1",
        "status": "partial",
        "results": [
            {
                "index": 0,
                "status": "failure",
                "error_type": "RuntimeError",
                "message": "unsupported element",
            }
        ],
    }

    summary = example.summarize_energy_results(payload, "batch-1", input_files)

    assert summary["all_succeeded"] is False
    assert summary["results_received"] == 1
    assert summary["failed"] == 2
    assert [failure["structure"] for failure in summary["failures"]] == [
        "a.xyz",
        "b.xyz",
    ]
    assert summary["failures"][1]["error_type"] == "MissingResult"


def test_summarize_energy_results_rejects_duplicate_indexes(example, tmp_path):
    payload = {
        "batch_id": "batch-1",
        "status": "completed",
        "results": [
            {"index": 0, "status": "success", "potential_energy": -1.0},
            {"index": 0, "status": "success", "potential_energy": -2.0},
        ],
    }

    with pytest.raises(RuntimeError, match="Duplicate Compute result index"):
        example.summarize_energy_results(
            payload,
            "batch-1",
            [tmp_path / "a.xyz", tmp_path / "b.xyz"],
        )


def test_report_failure_expands_exception_groups(example, capsys):
    error = ExceptionGroup("tool call failed", [RuntimeError("inner tool error")])

    example._report_failure(error)

    stderr = capsys.readouterr().err
    assert "FAIL: ExceptionGroup: tool call failed" in stderr
    assert "RuntimeError: inner tool error" in stderr


def test_deep_agent_trace_streams_messages_and_tools_once(example, capsys):
    initial_state = {
        "messages": [
            HumanMessage(content="Stage the structures."),
            AIMessage(content="I will inspect authorization=do-not-print."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "transfer_files",
                        "args": {"wait": True, "access_token": "do-not-print"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    completed_state = {
        "messages": [
            *initial_state["messages"],
            ToolMessage(
                name="transfer_files",
                tool_call_id="call-1",
                content=json.dumps({"status": "completed"}),
            ),
            _tool_call_message("read_file", {"file_path": "/large/result"}),
            _tool_message("read_file", {"content": "result preview"}),
            AIMessage(content="The workflow completed."),
        ]
    }

    seen = example.print_deep_agent_trace(initial_state)
    example.print_deep_agent_trace(completed_state, seen)
    example.print_deep_agent_trace(completed_state, seen)

    stdout = capsys.readouterr().out
    assert stdout.count("Deep Agent input [human]:") == 1
    assert stdout.count("Stage the structures.") == 1
    assert stdout.count("Deep Agent output [assistant]:") == 2
    assert stdout.count("The workflow completed.") == 1
    assert stdout.count("Deep Agent tool call [call-1]: transfer_files") == 1
    assert stdout.count("Deep Agent tool result [call-1]: transfer_files") == 1
    assert stdout.count(
        "Deep Agent tool call [read_file-call]: read_file"
    ) == 1
    assert stdout.count(
        "Deep Agent tool result [read_file-call]: read_file"
    ) == 1
    assert '"wait": true' in stdout
    assert '"status": "completed"' in stdout
    assert "[REDACTED]" in stdout
    assert "do-not-print" not in stdout


def test_deep_agent_trace_expands_offloaded_results_only_when_enabled(
    example,
    capsys,
):
    path = "/large_tool_results/call-1"
    payload = {
        "status": "completed",
        "results": [{"potential_energy": -14.03}],
        "access_token": "do-not-print",
    }
    message = ToolMessage(
        name="get_job_results",
        tool_call_id="call-1",
        content=(
            "Tool result too large and was saved in the filesystem at this "
            f'path: {path}. Preview: {{"access_token": "do-not-print"}}'
        ),
    )
    state = {
        "messages": [message],
        "files": {
            path: {"content": json.dumps(payload), "encoding": "utf-8"}
        },
    }

    example.print_deep_agent_trace(state)
    preview = capsys.readouterr().out
    assert path in preview
    assert "potential_energy" not in preview
    assert "[REDACTED]" in preview
    assert "do-not-print" not in preview

    example.print_deep_agent_trace(state, include_offloaded_payloads=True)
    expanded = capsys.readouterr().out
    assert '"potential_energy": -14.03' in expanded
    assert "[REDACTED]" in expanded
    assert "do-not-print" not in expanded


def test_find_tool_payload_recovers_offloaded_state_file(example):
    path = "/large_tool_results/call-1"
    payload = {
        "batch_id": "batch-1",
        "status": "completed",
        "results": [
            {"index": index, "status": "success", "potential_energy": -index}
            for index in range(166)
        ],
    }
    message = ToolMessage(
        name="get_job_results",
        tool_call_id="call-1",
        content=[
            {
                "type": "text",
                "text": (
                    "Tool result too large, the result of this tool call "
                    "call-1 was saved in the filesystem at this path: "
                    f"{path}\n\nUse read_file to inspect it."
                ),
            }
        ],
    )
    state = {
        "messages": [message],
        "files": {
            path: {"content": json.dumps(payload), "encoding": "utf-8"}
        },
    }

    assert example.find_tool_payload(state, "get_job_results") == payload


def test_decode_tool_payload_reports_missing_offloaded_file(example):
    content = (
        "Tool result too large, the result of this tool call call-1 was saved "
        "in the filesystem at this path: /large_tool_results/call-1"
    )

    with pytest.raises(ValueError, match="contains no matching file"):
        example.decode_tool_payload(content, files={})


@pytest.mark.parametrize(
    ("file_data", "error"),
    [
        ({"content": "e30=", "encoding": "base64"}, "unsupported encoding"),
        ({"content": "not json", "encoding": "utf-8"}, "valid JSON"),
    ],
)
def test_decode_tool_payload_rejects_invalid_offloaded_content(
    example,
    file_data,
    error,
):
    path = "/large_tool_results/call-1"
    content = f"saved in the filesystem at this path: {path}"

    with pytest.raises(ValueError, match=error):
        example.decode_tool_payload(content, files={path: file_data})


def test_validate_mace_tool_call_accepts_mace_and_rejects_emt(example):
    state = {
        "messages": [
            _tool_call_message(
                "run_ase_ensemble",
                {
                    "params": {
                        "calculator": {"calculator_type": "mace_mp"},
                    }
                },
            ),
        ]
    }

    example.validate_mace_tool_call(state)

    state["messages"] = [
        _tool_call_message(
            "run_ase_ensemble",
            {"params": {"calculator": {"calculator_type": "emt"}}},
        )
    ]
    with pytest.raises(RuntimeError, match="must use a MACE calculator"):
        example.validate_mace_tool_call(state)


def test_server_environment_forwards_port_but_not_llm_secret(example, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward")
    monkeypatch.setenv("GLOBUS_COMPUTE_ENDPOINT_ID", "endpoint-id")
    monkeypatch.setenv(
        "GLOBUS_TRANSFER_DESTINATION_COMPUTE_BASE_PATH",
        "/flare/MyProject/staging",
    )
    monkeypatch.setenv("CHEMGRAPH_REMOTE_DIRECTORY_TIMEOUT", "900")

    environment = example._server_environment(443)

    assert environment["GLOBUS_COMPUTE_AMQP_PORT"] == "443"
    assert environment["GLOBUS_COMPUTE_ENDPOINT_ID"] == "endpoint-id"
    assert environment["GLOBUS_TRANSFER_DESTINATION_COMPUTE_BASE_PATH"] == (
        "/flare/MyProject/staging"
    )
    assert environment["CHEMGRAPH_REMOTE_DIRECTORY_TIMEOUT"] == "900"
    assert environment["CHEMGRAPH_EXECUTION_BACKEND"] == "globus_compute"
    assert "OPENAI_API_KEY" not in environment


def test_trace_full_payloads_cli_flag_is_opt_in(example, monkeypatch):
    monkeypatch.delenv("GLOBUS_COMPUTE_AMQP_PORT", raising=False)

    assert example._parser().parse_args([]).trace_full_payloads is False
    assert (
        example._parser().parse_args(["--trace-full-payloads"]).trace_full_payloads
        is True
    )


def test_run_example_completes_workflow_in_one_agent_turn(
    example,
    monkeypatch,
    tmp_path,
    capsys,
):
    input_dir = tmp_path / "structures"
    input_dir.mkdir()
    for name in ("a.xyz", "b.xyz"):
        (input_dir / name).write_text(
            "3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n",
            encoding="utf-8",
        )
    loaded_tools = [_FakeTool(name) for name in example.BOUND_TOOL_NAMES]
    loaded_tools.append(_FakeTool("unrelated_tool"))
    graph_calls = []
    offloaded_path = "/large_tool_results/get-job-results-call"
    offloaded_result = {
        "status": "completed",
        "batch_id": "batch-1",
        "results": [
            {"index": 0, "status": "success", "potential_energy": -2.5},
            {"index": 1, "status": "success", "potential_energy": -1.5},
        ],
    }

    class FakeGraph:
        async def astream(self, inputs, *, config, stream_mode):
            graph_calls.append((inputs, config, stream_mode))
            final_state = {
                "messages": [
                    *inputs["messages"],
                    _tool_call_message("list_transfer_facilities", {}),
                    _tool_message(
                        "list_transfer_facilities",
                        {
                            "selection_mode": "server_configured",
                            "transfer_configured": True,
                            "active_system": "polaris",
                            "facilities": [],
                        },
                    ),
                    _tool_call_message("check_endpoint_status", {}),
                    _tool_message(
                        "check_endpoint_status",
                        {"status": {"status": "online"}},
                    ),
                    _tool_call_message(
                        "transfer_files",
                        {
                            "source_paths": str(input_dir.resolve()),
                            "extensions": [".xyz"],
                            "wait": True,
                        },
                    ),
                    _tool_message(
                        "transfer_files",
                        {
                            "status": "completed",
                            "remote_directory": "/remote/batch-1",
                            "transfer_directory": "/collection/batch-1",
                            "file_count": 2,
                        },
                    ),
                    _tool_call_message(
                        "run_ase_ensemble",
                        {
                            "params": {
                                "remote_structure_directory": "/remote/batch-1",
                                "output_results_file": "globus_ase_energy.json",
                                "driver": "energy",
                                "calculator": {
                                    "calculator_type": "mace_mp",
                                    "model": "small",
                                    "device": "cuda",
                                },
                            }
                        },
                    ),
                    _tool_message(
                        "run_ase_ensemble",
                        {
                            "status": "submitted",
                            "batch_id": "batch-1",
                            "n_tasks": 2,
                        },
                    ),
                    _tool_call_message(
                        "wait_for_job",
                        {
                            "batch_id": "batch-1",
                            "timeout": 1,
                            "poll_interval": 0.5,
                        },
                    ),
                    _tool_message(
                        "wait_for_job",
                        {"status": "completed", "batch_id": "batch-1"},
                    ),
                    _tool_call_message(
                        "get_job_results",
                        {"batch_id": "batch-1", "include_partial": True},
                    ),
                    ToolMessage(
                        name="get_job_results",
                        tool_call_id="get_job_results-call",
                        content=(
                            "Tool result too large, the result of this tool "
                            "call get_job_results-call was saved in the "
                            "filesystem at this path: "
                            f"{offloaded_path}"
                        ),
                    ),
                    AIMessage(
                        content="Completed 2 MACE calculations with no failures."
                    ),
                ],
                "files": {
                    offloaded_path: {
                        "content": json.dumps(offloaded_result),
                        "encoding": "utf-8",
                    }
                },
            }
            messages = final_state["messages"]
            for message_count in range(1, len(messages) + 1):
                state = {"messages": messages[:message_count]}
                if message_count == len(messages):
                    state["files"] = final_state["files"]
                yield state

    class FakeSessionContext:
        async def __aenter__(self):
            return "session"

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def session(self, name):
            assert name == "ChemGraph ASE (Globus)"
            return FakeSessionContext()

    async def fake_load_tools(session):
        assert session == "session"
        return loaded_tools

    captured = {}

    def fake_construct(model, **kwargs):
        captured.update(model=model, **kwargs)
        return FakeGraph()

    monkeypatch.setattr(example, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(example, "load_mcp_tools", fake_load_tools)
    monkeypatch.setattr(example, "load_chat_model", lambda **_kwargs: "model")
    monkeypatch.setattr(example, "construct_deep_agent_graph", fake_construct)
    args = argparse.Namespace(
        model="test:model",
        input=input_dir,
        amqp_port=443,
        compute_timeout=1,
        poll_interval=0.5,
        trace_full_payloads=False,
    )

    summary = asyncio.run(example.run_example(args))

    assert summary["all_succeeded"] is True
    assert summary["succeeded"] == 2
    assert [tool.name for tool in captured["tools"]] == list(
        example.BOUND_TOOL_NAMES
    )
    assert len(graph_calls) == 1
    request = graph_calls[0][0]["messages"][0].content
    assert request == (
        f"Stage every .xyz file in {input_dir.resolve()} to the configured HPC "
        "facility and run a MACE-MP small CUDA energy simulation over the "
        "staged structures. Complete the workflow and report the "
        "success/failure summary. Expect 2 input structure(s); wait up to 1 "
        "seconds and use a 0.5-second polling interval."
    )
    assert graph_calls[0][2] == "values"
    assert captured["system_prompt"] == example.GLOBUS_ASE_SYSTEM_PROMPT
    assert "transfer_files" not in captured["system_prompt"]
    stdout = capsys.readouterr().out
    assert "Deep Agent input [system]:" in stdout
    assert "Deep Agent input [human]:" in stdout
    assert stdout.count("Deep Agent tool call") == 6
    assert stdout.count("Deep Agent tool result") == 6
    assert "Deep Agent output [assistant]:" in stdout
    assert "Completed 2 MACE calculations with no failures." in stdout
