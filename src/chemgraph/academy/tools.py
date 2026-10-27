"""Peer messaging as an ordinary ChemGraph/LangChain tool."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated
from uuid import uuid4, uuid5

from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from pydantic import BaseModel, Field

from chemgraph.academy.message import PeerMessage

if TYPE_CHECKING:
    from academy.handle import Handle


class _SendPeerMessage(BaseModel):
    recipient: str
    content: str = Field(min_length=1)
    tool_call_id: Annotated[str, InjectedToolCallId]
    reply_to: str | None = None


def make_peer_message_tool(*, sender: str, peers: Mapping[str, "Handle"]) -> StructuredTool:
    """Use the supplied handles, with stable IDs for this tool instance's calls.

    Keep one tool instance per workflow/session. Repeating a tool-call ID uses
    the same message ID; a new tool instance gets a new namespace. Transport
    failures propagate: delivery may have occurred, so retries belong to the
    caller. No handles are registered, discovered, closed, or retried here.
    """
    namespace = uuid4()
    peers = dict(peers)

    async def send_peer_message(
        recipient: str, content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        reply_to: str | None = None,
    ) -> dict:
        if recipient not in peers:
            raise ToolException(f"Unknown peer {recipient!r}. Available peers: {list(peers)}")
        message = PeerMessage(
            message_id=str(uuid5(namespace, tool_call_id)),
            sender=sender, content=content, reply_to=reply_to,
        )
        acknowledgement = await peers[recipient].action("receive_message", message.model_dump(mode="json"))
        if acknowledgement != {"status": "accepted", "message_id": message.message_id}:
            raise RuntimeError("Invalid peer acknowledgement; delivery may have occurred")
        return acknowledgement

    return StructuredTool.from_function(
        coroutine=send_peer_message, name="send_peer_message", args_schema=_SendPeerMessage,
        handle_tool_error=True,
        description=(
            f"Send a request or reply to one of {list(peers)}. Acceptance means queued in memory, "
            "not completed. Send once and finish your turn to wait for a reply. "
            "For replies, set reply_to to the incoming message_id. Do not poll or acknowledge "
            "replies. Report to the human operator with a normal final assistant response."
        ),
    )
