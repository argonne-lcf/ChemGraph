from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

from chemgraph.academy_sim.config import AcademySimConfig, GraphConfig
from chemgraph.academy_sim.envelopes import build_envelope
from chemgraph.agent.graph_runtime import GraphRunResult


def _install_fake_academy(monkeypatch):
    class FakeAgent:
        pass

    def identity_decorator(fn):
        return fn

    academy = types.ModuleType("academy")
    agent = types.ModuleType("academy.agent")
    agent.Agent = FakeAgent
    agent.action = identity_decorator
    agent.loop = identity_decorator

    monkeypatch.setitem(sys.modules, "academy", academy)
    monkeypatch.setitem(sys.modules, "academy.agent", agent)
    sys.modules.pop("chemgraph.academy_sim.agent", None)


def _config_and_graph() -> tuple[AcademySimConfig, GraphConfig]:
    config = AcademySimConfig.model_validate(
        {
            "run_id": "run-1",
            "task": "test",
            "model": {"config_file": "lm.json"},
            "graphs": {
                "planner": {
                    "allowed_peers": ["executor"],
                },
                "executor": {},
            },
        }
    )
    return config, config.graph("planner")


@pytest.mark.asyncio
async def test_agent_marks_done_after_natural_final_answer(tmp_path, monkeypatch):
    _install_fake_academy(monkeypatch)
    agent_module = importlib.import_module("chemgraph.academy_sim.agent")
    config, graph = _config_and_graph()
    done_event = asyncio.Event()

    async def run_graph(_prompt: str) -> GraphRunResult:
        return GraphRunResult(output="final answer", terminal_tool=None)

    agent = agent_module.ChemGraphSimAgent(
        config=config,
        graph=graph,
        run_graph=run_graph,
        run_dir=tmp_path,
        done_event=done_event,
    )
    envelope = build_envelope(
        run_id="run-1",
        sender="executor",
        recipient="planner",
        content="results",
    )

    await agent._run_envelope(envelope)

    assert done_event.is_set()
    assert agent.last_output == "final answer"


@pytest.mark.asyncio
async def test_agent_waits_after_peer_send_terminal_turn(tmp_path, monkeypatch):
    _install_fake_academy(monkeypatch)
    agent_module = importlib.import_module("chemgraph.academy_sim.agent")
    config, graph = _config_and_graph()
    done_event = asyncio.Event()

    async def run_graph(_prompt: str) -> GraphRunResult:
        return GraphRunResult(
            output='{"status": "sent"}',
            executed_tool_names=("send_message_to_executor",),
            terminal_tool="send_message_to_executor",
        )

    agent = agent_module.ChemGraphSimAgent(
        config=config,
        graph=graph,
        run_graph=run_graph,
        run_dir=tmp_path,
        done_event=done_event,
    )
    envelope = build_envelope(
        run_id="run-1",
        sender="startup",
        recipient="planner",
        content="run task",
    )

    await agent._run_envelope(envelope)

    assert not done_event.is_set()
    assert agent.last_output == '{"status": "sent"}'
