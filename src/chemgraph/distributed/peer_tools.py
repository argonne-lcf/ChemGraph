"""Peer communication is a normal ChemGraph tool, independent of Academy."""

import json
import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from pydantic import BaseModel, Field

from chemgraph.distributed.protocol import PeerEnvelope, PeerTransport


class SendPeerMessage(BaseModel):
    recipient: str = Field(description="An allowed ChemGraph peer name. The operator receives normal final responses.")
    content: str = Field(min_length=1)
    tool_call_id: Annotated[str, InjectedToolCallId]
    correlation_id: str | None = None
    reply_to: str | None = None


def build_peer_tool(*, transport: PeerTransport, run_id: str, launch_id: str,
                    sender: str, allowed_peers: tuple[str, ...]) -> StructuredTool:
    async def send_peer_message(recipient: str, content: str,
                                tool_call_id: Annotated[str, InjectedToolCallId],
                                correlation_id: str | None = None,
                                reply_to: str | None = None) -> dict:
        if recipient not in allowed_peers:
            if recipient == "operator":
                raise ToolException(
                    "The operator is the human caller, not an addressable peer. "
                    "Return your result as a normal final assistant response; ChemGraph exposes it in peer status."
                )
            raise ToolException(f"Recipient {recipient!r} is not an allowed peer. Allowed peers: {list(allowed_peers)}.")
        identity = json.dumps(["chemgraph-peer-v2", run_id, launch_id, sender, tool_call_id])
        return await transport.send(PeerEnvelope(
            message_id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
            run_id=run_id, launch_id=launch_id, sender=sender, recipient=recipient,
            content=content, correlation_id=correlation_id, reply_to=reply_to,
        ))

    return StructuredTool.from_function(
        coroutine=send_peer_message, name="send_peer_message", args_schema=SendPeerMessage,
        handle_tool_error=True,
        description=(f"Send a message to one of {list(allowed_peers)}. Acceptance confirms delivery to an "
                     "in-memory inbox, not task completion. Send a request once, then finish your turn "
                     "to await a reply. Do not repeatedly poll peers. Reply using the incoming message ID "
                     "and correlation ID when provided. Sending a reply does not require another reply. "
                     "To report to the operator, return a normal final assistant response instead of using this tool."),
    )
