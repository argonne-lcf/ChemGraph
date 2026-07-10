from __future__ import annotations

import uuid

import pytest

from chemgraph.academy_sim.registrations import (
    deterministic_graph_agent_id,
    deterministic_graph_uid,
    load_agent_ids,
    publish_agent_id,
    wait_for_peer_uids,
)


class FakeAgentId:
    def __init__(self, value: str):
        self.value = value

    def model_dump(self, mode: str = "json"):
        return {"value": self.value}


def test_registration_filters_stale_launch_tokens(tmp_path, monkeypatch):
    class FakeAcademyAgentId:
        @classmethod
        def model_validate(cls, payload):
            return payload["value"]

        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    import types
    import sys

    fake_identifier = types.ModuleType("academy.identifier")
    fake_identifier.AgentId = FakeAcademyAgentId
    monkeypatch.setitem(sys.modules, "academy.identifier", fake_identifier)

    publish_agent_id(
        run_dir=tmp_path,
        run_id="run-1",
        run_token="token",
        launch_token="old",
        exchange_type="local",
        graph="planner",
        agent_id=FakeAgentId("old-planner"),
    )
    publish_agent_id(
        run_dir=tmp_path,
        run_id="run-1",
        run_token="token",
        launch_token="new",
        exchange_type="local",
        graph="executor",
        agent_id=FakeAgentId("new-executor"),
    )

    ids = load_agent_ids(tmp_path, run_token="token", launch_token="new")

    assert ids == {"executor": "new-executor"}


def test_registration_rejects_run_token_mismatch(tmp_path):
    publish_agent_id(
        run_dir=tmp_path,
        run_id="run-1",
        run_token="token",
        launch_token="launch",
        exchange_type="local",
        graph="planner",
        agent_id=FakeAgentId("planner"),
    )

    with pytest.raises(RuntimeError, match="different run token"):
        load_agent_ids(tmp_path, run_token="other")


def test_deterministic_graph_uid_is_stable_and_namespaced():
    uid = deterministic_graph_uid(run_id="run-1", graph_name="planner")

    assert uid == deterministic_graph_uid(run_id="run-1", graph_name="planner")
    assert uid != deterministic_graph_uid(run_id="run-2", graph_name="planner")
    assert uid != deterministic_graph_uid(run_id="run-1", graph_name="executor")


def test_deterministic_graph_agent_id_preserves_name(monkeypatch):
    class FakeAcademyAgentId:
        def __init__(self, *, uid, name, role):
            self.uid = uid
            self.name = name
            self.role = role

    import sys
    import types

    fake_identifier = types.ModuleType("academy.identifier")
    fake_identifier.AgentId = FakeAcademyAgentId
    monkeypatch.setitem(sys.modules, "academy.identifier", fake_identifier)

    agent_id = deterministic_graph_agent_id(run_id="run-1", graph_name="planner")

    assert agent_id.uid == deterministic_graph_uid(
        run_id="run-1",
        graph_name="planner",
    )
    assert agent_id.name == "planner"
    assert agent_id.role == "agent"


class _SeenAgentId:
    def __init__(self, uid, name=None):
        self.uid = uid
        self.name = name


class _FakeTransport:
    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = 0

    async def discover(self, _agent_class):
        index = min(self.calls, len(self.rounds) - 1)
        self.calls += 1
        return self.rounds[index]


@pytest.mark.asyncio
async def test_wait_for_peer_uids_matches_uid_when_names_are_missing():
    wanted = _SeenAgentId(uuid.uuid4(), name="executor")
    seen = _SeenAgentId(wanted.uid, name=None)
    transport = _FakeTransport([[seen]])

    await wait_for_peer_uids(
        transport,
        [wanted],
        agent_class=object,
        timeout_s=0.1,
        poll_interval_s=0.01,
    )

    assert transport.calls == 1


@pytest.mark.asyncio
async def test_wait_for_peer_uids_ignores_unrelated_agents_and_times_out():
    wanted = _SeenAgentId(uuid.uuid4(), name="executor")
    transport = _FakeTransport([[_SeenAgentId(uuid.uuid4(), name=None)]])

    with pytest.raises(TimeoutError, match="executor"):
        await wait_for_peer_uids(
            transport,
            [wanted],
            agent_class=object,
            timeout_s=0.02,
            poll_interval_s=0.01,
        )
