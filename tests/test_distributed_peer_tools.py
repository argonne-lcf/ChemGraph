from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chemgraph.distributed.peer_tools import build_peer_tool


@pytest.mark.asyncio
@pytest.mark.parametrize("recipient", ["operator", "unknown"])
async def test_invalid_recipient_is_a_recoverable_tool_error(recipient):
    transport = SimpleNamespace(send=AsyncMock())
    tool = build_peer_tool(transport=transport, run_id="run", launch_id="launch", sender="planner",
                           allowed_peers=("executor",))
    result = await tool.ainvoke({"name": tool.name, "id": "call", "type": "tool_call",
                                "args": {"recipient": recipient, "content": "result"}})
    assert result.status == "error"
    assert ("normal final assistant response" if recipient == "operator" else "executor") in result.content
    transport.send.assert_not_called()


@pytest.mark.asyncio
async def test_transport_failure_is_not_hidden_as_recipient_error():
    transport = SimpleNamespace(send=AsyncMock(side_effect=TimeoutError("delivery uncertain")))
    tool = build_peer_tool(transport=transport, run_id="run", launch_id="launch", sender="planner",
                           allowed_peers=("executor",))
    with pytest.raises(TimeoutError, match="delivery uncertain"):
        await tool.ainvoke({"recipient": "executor", "content": "task"})
    transport.send.assert_awaited_once()
