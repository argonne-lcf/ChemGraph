from __future__ import annotations

import pytest

from chemgraph.academy_sim.peer_tools import (
    build_peer_tools,
    peer_prompt_section,
    peer_tool_name,
)


class FakeHandle:
    def __init__(self):
        self.calls = []

    async def action(self, name, payload):
        self.calls.append((name, payload))
        return {"ok": True}


@pytest.mark.asyncio
async def test_peer_tool_sends_envelope():
    handle = FakeHandle()
    tools = build_peer_tools(
        run_id="run-1",
        sender="planner",
        allowed_peers=("executor",),
        peer_handles={"executor": handle},
    )

    result = await tools[0].ainvoke({"content": "please run water"})

    assert result["status"] == "sent"
    assert result["recipient"] == "executor"
    assert handle.calls[0][0] == "receive_message"
    assert handle.calls[0][1]["sender"] == "planner"
    assert handle.calls[0][1]["recipient"] == "executor"


def test_peer_prompt_lists_only_allowed_peers():
    text = peer_prompt_section(
        graph_name="planner",
        allowed_peers=("executor",),
    )

    assert "send_message_to_executor" in text
    assert "reviewer" not in text
    assert "do not send duplicate" in text
    assert "status 'sent' means delivery was accepted" in text


def test_peer_tool_description_discourages_polling_or_duplicate_dispatch():
    tools = build_peer_tools(
        run_id="run-1",
        sender="planner",
        allowed_peers=("executor",),
        peer_handles={"executor": FakeHandle()},
    )

    assert tools[0].name == peer_tool_name("executor")
    assert "send one complete request and then wait" in tools[0].description
    assert "poll for status" in tools[0].description
    assert "not task completion" in tools[0].description

    content = tools[0].args_schema.model_fields["content"]
    assert "send one complete request" in content.description

    correlation_id = tools[0].args_schema.model_fields["correlation_id"]
    assert "do not change it" in correlation_id.description


def test_peer_tools_require_discovered_handle():
    with pytest.raises(RuntimeError, match="no discovered Academy handle"):
        build_peer_tools(
            run_id="run-1",
            sender="planner",
            allowed_peers=("executor",),
            peer_handles={},
        )
