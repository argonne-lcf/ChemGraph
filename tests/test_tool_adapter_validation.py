from __future__ import annotations

import json
from typing import Any

import pytest

from chemgraph.academy.core.tools import build_chemgraph_reasoning_tools
from chemgraph.academy.core.campaign import ChemGraphAgentSpec


class _FakePeerHandle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def action(self, name: str, payload: dict[str, Any]) -> None:
        self.calls.append((name, payload))


def _agent_spec() -> ChemGraphAgentSpec:
    return ChemGraphAgentSpec(
        name="agent-a",
        role="Worker",
        mission="Use explicit tools only.",
        allowed_peers=("agent-b",),
        mcp_servers=(),
    )


async def _build_tools(tmp_path):
    traces: list[tuple[str, dict[str, Any]]] = []
    outbox: list[dict[str, Any]] = []
    peer_handle = _FakePeerHandle()
    tools = await build_chemgraph_reasoning_tools(
        spec=_agent_spec(),
        run_dir=tmp_path,
        peer_names=("agent-b",),
        peer_handles={"agent-b": peer_handle},
        outbox=outbox,
        tool_results=[],
        get_round_index=lambda: 1,
        set_final_result=lambda result: None,
        trace=lambda event, payload: traces.append((event, payload)),
    )
    return {
        "tools": {tool.name: tool for tool in tools},
        "traces": traces,
        "outbox": outbox,
        "peer_handle": peer_handle,
    }


@pytest.mark.asyncio
async def test_send_message_invalid_args_return_structured_tool_error(tmp_path) -> None:
    env = await _build_tools(tmp_path)

    result = await env["tools"]["send_message"].ainvoke(
        {
            "recipient": "agent-b",
            "tldr": "invalid confidence",
            "content": "content",
            "artifact_refs": [],
            "tool_result_ids": [],
            "reason": "exercise validation",
            "confidence": 1.5,
        }
    )

    assert result["status"] == "error"
    assert result["error_type"] == "invalid_tool_arguments"
    assert result["errors"][0]["field"] == "confidence"
    assert env["outbox"] == []
    assert env["peer_handle"].calls == []
    assert env["traces"] == [
        (
            "tool_call_failed",
            {
                "tool_name": "send_message",
                "status": "failed",
                "error": "invalid_tool_arguments",
                "error_type": "invalid_tool_arguments",
                "errors": result["errors"],
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_message_disallowed_recipient_does_not_deliver(tmp_path) -> None:
    env = await _build_tools(tmp_path)

    result = await env["tools"]["send_message"].ainvoke(
        {
            "recipient": "not-a-peer",
            "tldr": "wrong peer",
            "content": "content",
            "artifact_refs": [],
            "tool_result_ids": [],
            "reason": "exercise validation",
            "confidence": 0.8,
        }
    )

    assert result == {
        "status": "error",
        "tool_name": "send_message",
        "error": "disallowed_recipient",
        "error_type": "disallowed_recipient",
        "recipient": "not-a-peer",
        "allowed_peers": ["agent-b"],
    }
    assert env["outbox"] == []
    assert env["peer_handle"].calls == []
    assert env["traces"][0][0] == "tool_call_failed"
    assert env["traces"][0][1]["error_type"] == "disallowed_recipient"


@pytest.mark.asyncio
async def test_send_message_request_requires_tldr(tmp_path) -> None:
    env = await _build_tools(tmp_path)

    result = await env["tools"]["send_message"].ainvoke(
        {
            "recipient": "agent-b",
            "tldr": "",
            "content": "What happened?",
            "artifact_refs": [],
            "tool_result_ids": [],
            "reply_requested": True,
            "reason": "need a peer check",
            "confidence": 0.5,
        }
    )

    assert result["status"] == "error"
    assert result["error_type"] == "invalid_tool_arguments"
    assert result["errors"][0]["field"] == "tldr"
    assert env["outbox"] == []
    assert env["peer_handle"].calls == []


@pytest.mark.asyncio
async def test_send_message_reply_requested_marks_question(tmp_path) -> None:
    env = await _build_tools(tmp_path)

    result = await env["tools"]["send_message"].ainvoke(
        {
            "recipient": "agent-b",
            "tldr": "need status",
            "content": "Please send current status.",
            "artifact_refs": [],
            "tool_result_ids": [],
            "reply_requested": True,
            "reason": "the report needs the peer status",
            "confidence": 0.7,
        }
    )

    assert result["status"] == "sent"
    assert env["outbox"][0]["reply_requested"] is True
    assert env["outbox"][0]["kind"] == "question"


@pytest.mark.asyncio
async def test_valid_send_message_still_delivers(tmp_path) -> None:
    env = await _build_tools(tmp_path)

    result = await env["tools"]["send_message"].ainvoke(
        {
            "recipient": "agent-b",
            "tldr": "candidate ready",
            "content": "Candidate C1 has a usable artifact.",
            "artifact_refs": ["artifacts/c1.xyz"],
            "tool_result_ids": ["tool-1"],
            "reply_requested": False,
            "reason": "peer needs the result",
            "confidence": 0.9,
        }
    )

    assert result["status"] == "sent"
    assert result["recipient"] == "agent-b"
    assert len(env["outbox"]) == 1
    assert env["outbox"][0]["reply_requested"] is False
    assert env["peer_handle"].calls[0][0] == "receive_message"
    assert env["peer_handle"].calls[0][1]["message_id"] == result["message_id"]
    assert [event for event, _ in env["traces"]] == [
        "message_sent",
        "message_delivered",
    ]
    assert {
        json.loads(line)["message_id"]
        for line in tmp_path.joinpath("messages.jsonl").read_text().splitlines()
    } == {result["message_id"]}
