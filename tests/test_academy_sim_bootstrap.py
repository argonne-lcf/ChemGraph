from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_fake_academy(monkeypatch):
    class FakeAgent:
        pass

    def identity_decorator(fn):
        return fn

    class FakeAgentId:
        def __init__(self, *, uid, name, role):
            self.uid = uid
            self.name = name
            self.role = role

    class FakeHandle:
        calls = []

        def __init__(self, agent_id):
            self.agent_id = agent_id

        async def action(self, name, payload):
            self.calls.append((self.agent_id, name, payload))
            return {"ok": True}

    academy = types.ModuleType("academy")
    agent = types.ModuleType("academy.agent")
    agent.Agent = FakeAgent
    agent.action = identity_decorator
    agent.loop = identity_decorator
    handle = types.ModuleType("academy.handle")
    handle.Handle = FakeHandle
    identifier = types.ModuleType("academy.identifier")
    identifier.AgentId = FakeAgentId

    monkeypatch.setitem(sys.modules, "academy", academy)
    monkeypatch.setitem(sys.modules, "academy.agent", agent)
    monkeypatch.setitem(sys.modules, "academy.handle", handle)
    monkeypatch.setitem(sys.modules, "academy.identifier", identifier)
    sys.modules.pop("chemgraph.academy_sim.agent", None)
    sys.modules.pop("chemgraph.academy_sim.bootstrap", None)
    return FakeHandle


class _FakeClient:
    def __init__(self):
        self._transport = object()
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.entered = False


class _FakeFactory:
    def __init__(self):
        self.client = _FakeClient()

    async def create_user_client(self, *, name, start_listener):
        self.name = name
        self.start_listener = start_listener
        return self.client


@pytest.mark.asyncio
async def test_bootstrap_dispatches_startup_envelope(tmp_path, monkeypatch):
    fake_handle = _install_fake_academy(monkeypatch)
    bootstrap = importlib.import_module("chemgraph.academy_sim.bootstrap")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "run_id": "run-1",
          "task": "fallback task",
          "initial_graph": "planner",
          "bootstrap_mode": "manual",
          "exchange": {"type": "http", "registration": "exchange"},
          "model": {"config_file": "lm.json"},
          "graphs": {
            "planner": {
              "startup_prompt": "start here",
              "allowed_peers": []
            }
          }
        }
        """,
        encoding="utf-8",
    )
    config = bootstrap.load_config(config_path)
    factory = _FakeFactory()
    waited = {}

    async def fake_wait(transport, peer_ids, *, agent_class, timeout_s):
        waited["transport"] = transport
        waited["peer_ids"] = list(peer_ids)
        waited["timeout_s"] = timeout_s

    monkeypatch.setattr(bootstrap, "build_exchange_factory", lambda _config: factory)
    monkeypatch.setattr(bootstrap, "wait_for_peer_uids", fake_wait)

    message_id = await bootstrap.dispatch_bootstrap(
        config=config,
        recipient="planner",
        timeout_s=3.0,
    )

    assert message_id.startswith("cg-peer-")
    assert waited["timeout_s"] == 3.0
    assert waited["peer_ids"][0].name == "planner"
    assert fake_handle.calls[0][1] == "receive_message"
    assert fake_handle.calls[0][2]["content"] == "start here"
    assert fake_handle.calls[0][2]["sender"] == "startup"
