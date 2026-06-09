from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from chemgraph.academy.core.agent import ChemGraphLogicalAgent
from chemgraph.academy.core.turn import (
    build_peer_status,
    ChemGraphReasoningRoundEngine,
)
from chemgraph.academy.core.turn import ReasoningTurnResult
from chemgraph.academy.core.tools import ReasoningToolRuntimeState
from chemgraph.academy.core.tools import build_chemgraph_reasoning_tools
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import ResourceSpec
from chemgraph.academy.core.campaign import resolve_campaign_resources
from chemgraph.academy.core.lm import LLMSettings
from chemgraph.academy.core.prompt import PromptProfile
from chemgraph.academy.core.prompt import PromptStateLimits


def _agent_spec() -> ChemGraphAgentSpec:
    return ChemGraphAgentSpec(
        name="agent-a",
        role="Worker",
        mission="Use explicit tools only.",
        allowed_peers=(),
        tools=(),
    )


def _agent_spec_with_peer() -> ChemGraphAgentSpec:
    return ChemGraphAgentSpec(
        name="agent-a",
        role="Worker",
        mission="Use explicit tools only.",
        allowed_peers=("agent-b",),
        tools=(),
    )


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


class _FakeReasoningEngine:
    async def run_turn(self) -> ReasoningTurnResult:
        return ReasoningTurnResult(
            final_text="done",
            state={"messages": []},
            tool_calls_completed=1,
            action_tools_called=("finish_turn",),
            science_tools_called=("science_tool",),
            executed_tool_names=("science_tool", "finish_turn"),
            requested_finish=True,
            requested_self_wake=True,
            workflow_span_id="workflow-1",
            thread_id="agent-a-round-1",
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
async def test_reasoning_adapter_finish_turn_updates_runtime_state(tmp_path) -> None:
    spec = _agent_spec()
    runtime_state = ReasoningToolRuntimeState()
    traces: list[tuple[str, dict]] = []

    tools = await build_chemgraph_reasoning_tools(
        spec=spec,
        run_dir=tmp_path,
        tool_invoker=object(),  # unused when spec.tools is empty
        peer_names=(),
        peer_handles={},
        outbox=[],
        tool_results=[],
        get_round_index=lambda: 1,
        set_final_result=lambda result: None,
        trace=lambda event, payload: traces.append((event, payload)),
        runtime_state=runtime_state,
    )

    assert [tool.name for tool in tools] == [
        "send_message",
        "ask_peer",
        "submit_result",
        "finish_turn",
    ]

    finish_turn = next(tool for tool in tools if tool.name == "finish_turn")
    result = await finish_turn.ainvoke({"reason": "nothing useful now"})

    assert result == {"status": "finished", "reason": "nothing useful now"}
    assert runtime_state.finished_turn is True
    assert runtime_state.action_tool_names == ["finish_turn"]
    assert runtime_state.executed_tool_names == ["finish_turn"]
    assert traces == [
        (
            "turn_finished_without_external_action",
            {"reason": "nothing useful now"},
        )
    ]


@pytest.mark.asyncio
async def test_send_message_does_not_block_on_busy_peer(tmp_path) -> None:
    spec = _agent_spec_with_peer()
    runtime_state = ReasoningToolRuntimeState()
    peer = _SlowPeerHandle()
    traces: list[tuple[str, dict]] = []
    outbox: list[dict] = []

    tools = await build_chemgraph_reasoning_tools(
        spec=spec,
        run_dir=tmp_path,
        tool_invoker=object(),
        peer_names=("agent-b",),
        peer_handles={"agent-b": peer},
        outbox=outbox,
        tool_results=[],
        get_round_index=lambda: 1,
        set_final_result=lambda result: None,
        trace=lambda event, payload: traces.append((event, payload)),
        runtime_state=runtime_state,
    )
    send_message = next(tool for tool in tools if tool.name == "send_message")

    result = await asyncio.wait_for(
        send_message.ainvoke(
            {
                "recipient": "agent-b",
                "tldr": "short summary",
                "content": "full message",
                "artifact_refs": [],
                "tool_result_ids": [],
                "reason": "peer needs this evidence",
                "confidence": 0.8,
            },
        ),
        timeout=0.05,
    )

    assert result["status"] == "sent"
    assert result["delivery"] == "queued"
    assert len(outbox) == 1
    assert [name for name, _ in traces] == ["message_sent"]

    await asyncio.wait_for(peer.delivered.wait(), timeout=1)
    await asyncio.sleep(0)

    assert peer.calls[0][0] == "receive_message"
    assert [name for name, _ in traces] == [
        "message_sent",
        "message_delivered",
    ]


@pytest.mark.asyncio
async def test_logical_agent_startup_initializes_chemgraph_reasoning_engine(
    tmp_path,
) -> None:
    spec = _agent_spec()
    agent = ChemGraphLogicalAgent(
        spec,
        campaign=_campaign(spec),
        llm_settings=_lm_settings(),
        prompt_profile=_prompt_profile(),
        run_dir=tmp_path,
        max_decisions=5,
        tool_invoker=object(),  # unused when spec.tools is empty
    )

    await agent.agent_on_startup()

    assert isinstance(agent._reasoning_engine, ChemGraphReasoningRoundEngine)


@pytest.mark.asyncio
async def test_logical_agent_reasoning_round_uses_chemgraph_engine(tmp_path) -> None:
    spec = _agent_spec()
    agent = ChemGraphLogicalAgent(
        spec,
        campaign=_campaign(spec),
        llm_settings=_lm_settings(),
        prompt_profile=_prompt_profile(),
        run_dir=tmp_path,
        max_decisions=5,
        tool_invoker=object(),
    )
    agent.round_index = 1
    agent._reasoning_engine = _FakeReasoningEngine()

    self_wake = await agent._reasoning_round()

    assert self_wake is True
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


def test_reasoning_engine_builds_bounded_wakeup_state(tmp_path) -> None:
    spec = _agent_spec()
    received_message_history = [{"message_id": "old"}, {"message_id": "new"}]
    outbox = [
        {
            "message_id": "msg-old",
            "recipient": "agent-b",
            "tldr": "old message",
            "timestamp": 1,
        },
        {
            "message_id": "msg-new",
            "recipient": "agent-b",
            "tldr": "new message",
            "timestamp": 3,
        },
    ]
    tool_results = [{"tool_result_id": "old"}, {"tool_result_id": "new"}]
    final_result = {"summary": "current belief"}
    engine = ChemGraphReasoningRoundEngine(
        campaign=_campaign(spec),
        spec=spec,
        llm_settings=_lm_settings(),
        prompt_profile=_prompt_profile(),
        run_dir=tmp_path,
        max_decisions=5,
        tools=[],
        runtime_state=ReasoningToolRuntimeState(),
        received_message_history=received_message_history,
        outbox=outbox,
        tool_results=tool_results,
        get_final_result=lambda: final_result,
        get_round_index=lambda: 2,
        trace=lambda event, payload: None,
    )

    state = engine.build_wakeup_state(round_index=2)

    assert state["campaign"] == "campaign-1"
    assert state["user_task"] == "Rank staged candidates."
    assert state["agent_name"] == "agent-a"
    assert state["available_chemgraph_tools"] == []
    assert state["peer_status"] == {}
    assert state["received_messages"] == [{"message_id": "new"}]
    assert state["local_chemgraph_tool_results"] == [{"tool_result_id": "new"}]
    assert state["recent_actions"] == [
        {
            "type": "send_message",
            "recipient": "agent-b",
            "tldr": "old message",
            "message_id": "msg-old",
            "timestamp": 1,
        },
        {
            "type": "send_message",
            "recipient": "agent-b",
            "tldr": "new message",
            "message_id": "msg-new",
            "timestamp": 3,
        },
    ]
    assert state["current_final_result"] == final_result
    assert state["required_protocol"] == "call finish_turn when idle"


def test_build_peer_status_uses_inflight_tool_events(tmp_path) -> None:
    state_dir = tmp_path / "agent_status"
    state_dir.mkdir()
    (state_dir / "agent-b.json").write_text(
        json.dumps(
            {
                "agent_name": "agent-b",
                "round": 3,
                "finished": False,
                "last_error": None,
                "status_updated_at": 100.0,
                "recent_outbox": [
                    {
                        "message_id": "msg-ack",
                        "tldr": "Starting requested MACE energy run",
                    },
                ],
                "belief": {
                    "hypothesis": None,
                    "confidence": 0.0,
                },
            },
        )
        + "\n",
        encoding="utf-8",
    )
    events = [
        {
            "timestamp": 101.0,
            "event": "message_sent",
            "agent_id": "agent-b",
            "payload": {
                "message_id": "msg-ack",
                "tldr": "Starting requested MACE energy run",
            },
        },
        {
            "timestamp": 102.0,
            "event": "tool_call_started",
            "agent_id": "agent-b",
            "payload": {
                "tool_name": "run_mace_ensemble",
                "tool_result_id": "tool-1",
                "tool_call_id": "call-1",
            },
        },
    ]
    with (tmp_path / "events.jsonl").open("w", encoding="utf-8") as fp:
        for event in events:
            fp.write(json.dumps(event) + "\n")

    status = build_peer_status(run_dir=tmp_path, peer_names=("agent-b",))

    assert status["agent-b"]["state"] == "busy"
    assert status["agent-b"]["last_outbox_tldr"] == "Starting requested MACE energy run"
    assert status["agent-b"]["current_activity"] == {
        "type": "tool_call",
        "tool_name": "run_mace_ensemble",
        "tool_result_id": "tool-1",
        "tool_call_id": "call-1",
        "started_at": 102.0,
    }


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

    assert campaign.resources["structure_output_directory"].path == (
        "academy_mace_structures"
    )
    assert resolved.resources["candidate_dataset"].path == (
        "/source/data/candidates.json"
    )
    assert resolved.resources["structure_output_directory"].path == str(
        tmp_path / "run-1" / "shared" / "academy_mace_structures",
    )
    assert resolved.resources["mace_output_result_file"].path == str(
        tmp_path / "run-1" / "shared" / "academy_mace_outputs" / "mace_results.json",
    )
