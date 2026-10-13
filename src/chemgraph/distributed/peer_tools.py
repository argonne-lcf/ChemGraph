"""Peer communication is a normal ChemGraph tool, independent of Academy."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chemgraph.distributed.protocol import PeerEnvelope, PeerTransport


class SendPeerMessage(BaseModel):
    recipient: str
    content: str = Field(min_length=1)
    correlation_id: str | None = None
    reply_to: str | None = None


def build_peer_tool(*, transport: PeerTransport, run_id: str, launch_id: str,
                    sender: str, allowed_peers: tuple[str, ...]) -> StructuredTool:
    async def send_peer_message(recipient: str, content: str, correlation_id: str | None = None,
                                reply_to: str | None = None) -> dict:
        if recipient not in allowed_peers:
            raise ValueError(f"Recipient {recipient!r} is not an allowed peer")
        return await transport.send(PeerEnvelope(
            run_id=run_id, launch_id=launch_id, sender=sender, recipient=recipient,
            content=content, correlation_id=correlation_id, reply_to=reply_to,
        ))

    return StructuredTool.from_function(
        coroutine=send_peer_message, name="send_peer_message", args_schema=SendPeerMessage,
        description=(f"Send a message to one of {list(allowed_peers)}. Acceptance confirms delivery to an "
                     "in-memory inbox, not task completion. Send a request once, then finish your turn "
                     "to await a reply. Do not repeatedly poll peers. Reply using the incoming message ID "
                     "and correlation ID when provided. Sending a reply does not require another reply."),
    )
