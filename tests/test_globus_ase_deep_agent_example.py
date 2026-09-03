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
from langchain_core.messages import ToolMessage


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


def test_wait_for_batch_polls_to_completion(example):
    tool = _FakeTool(
        "check_job_status",
        responses=[
            {"status": "pending", "total_tasks": 1, "completed_tasks": 0},
            {"status": "completed", "total_tasks": 1, "completed_tasks": 1},
        ],
    )

    result = asyncio.run(
        example.wait_for_batch(
            tool,
            "batch-1",
            timeout=1,
            poll_interval=0,
        )
    )

    assert result["status"] == "completed"
    assert tool.calls == [{"batch_id": "batch-1"}, {"batch_id": "batch-1"}]


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "failed", "failed_tasks": 1},
        {"status": "partial", "failed_tasks": 1},
        {"error": "unknown batch"},
    ],
)
def test_wait_for_batch_rejects_failures(example, payload):
    tool = _FakeTool("check_job_status", responses=[payload])

    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(
            example.wait_for_batch(
                tool,
                "batch-1",
                timeout=1,
                poll_interval=0,
            )
        )


def test_validate_energy_result_requires_one_finite_success(example):
    payload = {
        "batch_id": "batch-1",
        "status": "completed",
        "results": [{"status": "success", "potential_energy": -1.25}],
    }

    assert example.validate_energy_result(payload, "batch-1") == -1.25
    payload["results"][0]["potential_energy"] = float("nan")
    with pytest.raises(RuntimeError, match="not finite"):
        example.validate_energy_result(payload, "batch-1")


def test_server_environment_forwards_port_but_not_llm_secret(example, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward")
    monkeypatch.setenv("GLOBUS_COMPUTE_ENDPOINT_ID", "endpoint-id")

    environment = example._server_environment(443)

    assert environment["GLOBUS_COMPUTE_AMQP_PORT"] == "443"
    assert environment["GLOBUS_COMPUTE_ENDPOINT_ID"] == "endpoint-id"
    assert environment["CHEMGRAPH_EXECUTION_BACKEND"] == "globus_compute"
    assert "OPENAI_API_KEY" not in environment


def test_run_example_submits_polls_and_resumes_same_graph(
    example,
    monkeypatch,
    tmp_path,
):
    input_file = tmp_path / "water.xyz"
    input_file.write_text("3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", encoding="utf-8")
    status_tool = _FakeTool(
        "check_job_status",
        responses=[
            {"status": "pending", "total_tasks": 1, "completed_tasks": 0},
            {"status": "completed", "total_tasks": 1, "completed_tasks": 1},
        ],
    )
    loaded_tools = [
        status_tool if name == "check_job_status" else _FakeTool(name)
        for name in example.BOUND_TOOL_NAMES
    ]
    loaded_tools.append(_FakeTool("unrelated_tool"))
    graph_calls = []

    class FakeGraph:
        async def ainvoke(self, inputs, config):
            graph_calls.append((inputs, config))
            if len(graph_calls) == 1:
                return {
                    "messages": [
                        _tool_message(
                            "check_endpoint_status",
                            {"status": {"status": "online"}},
                        ),
                        _tool_message(
                            "transfer_files",
                            {
                                "status": "completed",
                                "remote_directory": "/remote/batch-1",
                            },
                        ),
                        _tool_message(
                            "run_ase_ensemble",
                            {"status": "submitted", "batch_id": "batch-1"},
                        ),
                    ]
                }
            return {
                "messages": [
                    _tool_message(
                        "get_job_results",
                        {
                            "status": "completed",
                            "batch_id": "batch-1",
                            "results": [
                                {
                                    "status": "success",
                                    "potential_energy": -2.5,
                                }
                            ],
                        },
                    )
                ]
            }

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
        input=input_file,
        amqp_port=443,
        compute_timeout=1,
        poll_interval=0,
    )

    energy = asyncio.run(example.run_example(args))

    assert energy == -2.5
    assert [tool.name for tool in captured["tools"]] == list(
        example.BOUND_TOOL_NAMES
    )
    assert len(graph_calls) == 2
    assert graph_calls[0][1] == graph_calls[1][1]
    assert "run_ase_ensemble" in graph_calls[0][0]["messages"][0].content
    assert "batch-1" in graph_calls[1][0]["messages"][0].content
    assert status_tool.calls == [
        {"batch_id": "batch-1"},
        {"batch_id": "batch-1"},
    ]
