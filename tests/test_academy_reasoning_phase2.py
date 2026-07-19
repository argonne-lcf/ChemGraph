from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

# Skip when the optional 'academy' extra is absent.
pytest.importorskip("academy")

from chemgraph.academy.core import agent as agent_module
from chemgraph.academy.core import turn as turn_module
from chemgraph.academy.core.agent import ChemGraphLogicalAgent
from chemgraph.academy.core.campaign import ChemGraphAgentSpec, ChemGraphCampaign
from chemgraph.academy.core.campaign import ResourceSpec, resolve_campaign_resources
from chemgraph.academy.core.prompt import PromptProfile, PromptStateLimits
from chemgraph.academy.core.tools import build_chemgraph_reasoning_tools
from chemgraph.academy.core.turn import ReasoningTurnResult, build_peer_status
from chemgraph.agent.turn import TurnResult
from chemgraph.models.settings import LLMSettings


def _agent_spec() -> ChemGraphAgentSpec:
    return ChemGraphAgentSpec(
        name="agent-a",
        role="Worker",
        mission="Use explicit tools only.",
        allowed_peers=(),
        mcp_servers=(),
    )


def _agent_spec_with_peer() -> ChemGraphAgentSpec:
    return dataclasses.replace(_agent_spec(), allowed_peers=("agent-b",))


def _campaign(spec: ChemGraphAgentSpec) -> ChemGraphCampaign:
    return ChemGraphCampaign(
        run_id="campaign-1",
        user_task="Rank staged candidates.",
        initial_agent=spec.name,
        prompt_profile=Path("prompt_profiles/default.json"),
        agents=(spec,),
    )


def _prompt_profile() -> PromptProfile:
    return PromptProfile(
        prompt_version="test",
        prompt_style="json_state",
        system_prompt="system prompt",
        protocol_prompt="call finish_turn when idle",
        langchain_recursion_limit=8,
        state_limits=PromptStateLimits(
            received_messages_last_n=1,
            tool_results_last_n=1,
            actions_last_n=2,
        ),
    )


def _lm_settings() -> LLMSettings:
    return LLMSettings(
        base_url="http://127.0.0.1:18085/argoapi/v1",
        model="GPT-5.4",
        provider="openai_compatible_tools",
        api_key="dummy",
        user="test-user",
        timeout_s=60,
        temperature=0,
        max_tokens=1024,
        max_retries=1,
        retry_delay_s=0,
    )


class _SlowPeerHandle:
    def __init__(self) -> None:
        self.delivered = asyncio.Event()
        self.calls: list[tuple[str, dict]] = []

    async def action(self, name: str, message: dict) -> None:
        await asyncio.sleep(0.1)
        self.calls.append((name, message))
        self.delivered.set()


@pytest.mark.asyncio
async def test_reasoning_adapter_finish_turn_traces(tmp_path) -> None:
    traces: list[tuple[str, dict]] = []
    tools = await build_chemgraph_reasoning_tools(
        spec=_agent_spec(),
        run_dir=tmp_path,
        peer_names=(),
        peer_handles={},
        outbox=[],
        tool_results=[],
        get_round_index=lambda: 1,
        set_final_result=lambda result: None,
        trace=lambda event, payload: traces.append((event, payload)),
    )

    result = await next(t for t in tools if t.name == "finish_turn").ainvoke(
        {"reason": "nothing useful now"},
    )

    assert result == {"status": "finished", "reason": "nothing useful now"}
    assert traces == [
        (
            "turn_finished_without_external_action",
            {"reason": "nothing useful now"},
        )
    ]


@pytest.mark.asyncio
async def test_send_message_does_not_block_on_busy_peer(tmp_path) -> None:
    peer = _SlowPeerHandle()
    traces: list[tuple[str, dict]] = []
    outbox: list[dict] = []
    tools = await build_chemgraph_reasoning_tools(
        spec=_agent_spec_with_peer(),
        run_dir=tmp_path,
        peer_names=("agent-b",),
        peer_handles={"agent-b": peer},
        outbox=outbox,
        tool_results=[],
        get_round_index=lambda: 1,
        set_final_result=lambda result: None,
        trace=lambda event, payload: traces.append((event, payload)),
    )

    result = await asyncio.wait_for(
        next(t for t in tools if t.name == "send_message").ainvoke(
            {
                "recipient": "agent-b",
                "tldr": "short summary",
                "content": "full message",
                "artifact_refs": [],
                "tool_result_ids": [],
                "reply_requested": False,
                "reason": "peer needs this evidence",
                "confidence": 0.8,
            },
        ),
        timeout=0.05,
    )

    assert result["delivery"] == "queued"
    assert len(outbox) == 1
    assert [name for name, _ in traces] == ["message_sent"]
    await asyncio.wait_for(peer.delivered.wait(), timeout=1)
    assert peer.calls[0][0] == "receive_message"


@pytest.mark.asyncio
async def test_run_academy_turn_maps_action_and_science_tools(monkeypatch, tmp_path) -> None:
    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        payload = json.loads(kwargs["query"])
        assert payload["received_messages"] == [{"message_id": "new"}]
        assert payload["local_chemgraph_tool_results"] == [{"tool_result_id": "new"}]
        kwargs["on_event"]("workflow_started", {"thread_id": kwargs["thread_id"]})
        return TurnResult(
            final_text="done",
            state={"messages": []},
            executed_tool_names=("science_tool", "finish_turn"),
            terminal_tool="finish_turn",
            thread_id=kwargs["thread_id"],
            duration_s=0.1,
        )

    monkeypatch.setattr(turn_module, "run_turn", fake_run_turn)
    traces: list[tuple[str, dict]] = []
    result = await turn_module.run_academy_turn(
        campaign=_campaign(_agent_spec()),
        spec=_agent_spec(),
        llm_settings=_lm_settings(),
        prompt_profile=_prompt_profile(),
        run_dir=tmp_path,
        max_decisions=5,
        tools=[],
        received_message_history=[{"message_id": "old"}, {"message_id": "new"}],
        outbox=[],
        tool_results=[{"tool_result_id": "old"}, {"tool_result_id": "new"}],
        get_final_result=lambda: {"summary": "current"},
        get_round_index=lambda: 2,
        trace=lambda event, payload: traces.append((event, payload)),
    )

    assert result.action_tools_called == ("finish_turn",)
    assert result.science_tools_called == ("science_tool",)
    assert result.requested_finish is True
    assert result.requested_self_wake is True
    assert [event for event, _ in traces] == [
        "chemgraph_reasoning_turn_started",
        "workflow_started",
        "chemgraph_reasoning_turn_finished",
    ]


@pytest.mark.asyncio
async def test_logical_agent_reasoning_round_calls_turn_runner(monkeypatch, tmp_path) -> None:
    spec = _agent_spec()
    agent = ChemGraphLogicalAgent(
        spec,
        campaign=_campaign(spec),
        llm_settings=_lm_settings(),
        prompt_profile=_prompt_profile(),
        run_dir=tmp_path,
        max_decisions=5,
    )
    agent.round_index = 1

    async def fake_tools(**kwargs: Any) -> list:
        assert kwargs["spec"] is spec
        return []

    async def fake_turn(**kwargs: Any) -> ReasoningTurnResult:
        assert kwargs["spec"] is spec
        return ReasoningTurnResult(
            final_text="done",
            executed_tool_names=("science_tool", "finish_turn"),
            action_tools_called=("finish_turn",),
            science_tools_called=("science_tool",),
            requested_finish=True,
            requested_self_wake=True,
            thread_id="agent-a-round-1",
        )

    monkeypatch.setattr(agent_module, "build_chemgraph_reasoning_tools", fake_tools)
    monkeypatch.setattr(agent_module, "run_academy_turn", fake_turn)

    assert await agent._reasoning_round() is True
    events = [
        json.loads(line)["event"]
        for line in tmp_path.joinpath("events.jsonl").read_text().splitlines()
    ]
    assert events == [
        "round_started",
        "agent_decision",
        "round_finished",
        "self_wake_scheduled",
    ]


def test_build_peer_status_uses_agent_status_file(tmp_path) -> None:
    state_dir = tmp_path / "agent_status"
    state_dir.mkdir()
    (state_dir / "agent-b.json").write_text(
        json.dumps(
            {
                "round": 3,
                "finished": False,
                "last_error": None,
                "status_updated_at": 100.0,
            },
        )
        + "\n",
        encoding="utf-8",
    )

    status = build_peer_status(run_dir=tmp_path, peer_names=("agent-b",))

    assert status["agent-b"]["state"] == "idle"
    assert status["agent-b"]["round"] == 3
    assert status["agent-b"]["last_error"] is None


def test_campaign_resources_resolve_to_shared_run_artifacts(tmp_path) -> None:
    spec = dataclasses.replace(
        _agent_spec(),
        resources=("candidate_dataset", "structure_output_directory"),
    )
    campaign = ChemGraphCampaign(
        run_id="campaign-1",
        user_task="Rank staged candidates.",
        initial_agent=spec.name,
        prompt_profile=Path("prompt_profiles/default.json"),
        agents=(spec,),
        resources={
            "candidate_dataset": ResourceSpec(
                kind="json",
                path="/source/data/candidates.json",
                scope="absolute",
                expose_content=True,
            ),
            "structure_output_directory": ResourceSpec(
                kind="directory",
                path="academy_mace_structures",
                scope="shared_run",
            ),
            "mace_output_result_file": ResourceSpec(
                kind="file",
                path="academy_mace_outputs/mace_results.json",
                scope="shared_run",
            ),
        },
    )

    resolved = resolve_campaign_resources(campaign, tmp_path / "run-1")

    assert resolved.resources["candidate_dataset"].path == "/source/data/candidates.json"
    assert resolved.resources["structure_output_directory"].path == str(
        tmp_path / "run-1" / "shared" / "academy_mace_structures",
    )
    assert resolved.resources["mace_output_result_file"].path == str(
        tmp_path / "run-1" / "shared" / "academy_mace_outputs" / "mace_results.json",
    )

    # The directory resource itself is materialised on disk so tools that
    # expect to write into it do not hit FileNotFoundError on first use.
    assert (
        tmp_path / "run-1" / "shared" / "academy_mace_structures"
    ).is_dir()
    # File resources get their parent directory materialised (the file
    # itself is the agent's responsibility to write).
    assert (
        tmp_path / "run-1" / "shared" / "academy_mace_outputs"
    ).is_dir()
    assert not (
        tmp_path / "run-1" / "shared" / "academy_mace_outputs" / "mace_results.json"
    ).exists()


def test_resolve_campaign_resources_skips_non_shared_run_paths(tmp_path) -> None:
    """Only shared_run resources get on-disk materialisation."""
    spec = dataclasses.replace(_agent_spec(), resources=("local_dataset",))
    campaign = ChemGraphCampaign(
        run_id="campaign-2",
        user_task="Static dataset.",
        initial_agent=spec.name,
        prompt_profile=Path("prompt_profiles/default.json"),
        agents=(spec,),
        resources={
            "local_dataset": ResourceSpec(
                kind="json",
                path="/should/not/exist/data.json",
                scope="absolute",
            ),
        },
    )

    resolved = resolve_campaign_resources(campaign, tmp_path / "run-1")

    # The absolute path is preserved verbatim and no directory is created.
    assert resolved.resources["local_dataset"].path == "/should/not/exist/data.json"
    assert not Path("/should/not/exist").exists()
