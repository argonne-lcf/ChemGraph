"""Academy-backed peer communication tools for ChemGraph graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from chemgraph.academy_sim.envelopes import build_envelope


class SendPeerMessageInput(BaseModel):
    """Input schema for generated peer-send tools."""

    content: str = Field(
        min_length=1,
        description=(
            'Message content to send to the peer graph. For a delegated task, '
            'send one complete request and then wait for that peer to reply.'
        ),
    )
    correlation_id: str | None = Field(
        default=None,
        description=(
            'Optional stable correlation id for a multi-message task. Reuse it '
            'when replying to an actual peer response; do not change it to '
            'create duplicate canonical or status-poll requests.'
        ),
    )
    reply_to: str | None = Field(
        default=None,
        description='Optional message id this message replies to.',
    )


def peer_prompt_section(
    *,
    graph_name: str,
    allowed_peers: tuple[str, ...],
) -> str:
    """Return prompt text describing graph-to-graph communication tools."""

    if not allowed_peers:
        return (
            "\n\nGraph-to-graph communication: no peer graphs are available "
            f"to {graph_name}."
        )
    tool_lines = [
        f"- {peer}: use {peer_tool_name(peer)}"
        for peer in allowed_peers
    ]
    return (
        "\n\nGraph-to-graph communication:\n"
        f"You are graph {graph_name}. You may communicate only with these "
        "peer graphs:\n"
        + "\n".join(tool_lines)
        + "\nUse peer tools only when another graph should act, compute, "
        "clarify, or receive a final result. After sending a peer request for "
        "a task, do not send duplicate, canonical, final, acknowledgement, or "
        "status-check messages for the same task unless you first receive a "
        "peer reply or a new human instruction. A successful peer-tool result "
        "with status 'sent' means delivery was accepted; it is not the "
        "scientific result."
    )


def build_peer_tools(
    *,
    run_id: str,
    sender: str,
    allowed_peers: tuple[str, ...],
    peer_handles: Mapping[str, Any],
    trace: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[BaseTool]:
    """Build one send tool per allowed peer."""

    trace = trace or (lambda _event, _payload: None)
    tools: list[BaseTool] = []
    for recipient in allowed_peers:
        if recipient not in peer_handles:
            raise RuntimeError(
                f'allowed peer {recipient!r} has no discovered Academy handle'
            )
        handle = peer_handles[recipient]
        tools.append(
            _build_one_peer_tool(
                run_id=run_id,
                sender=sender,
                recipient=recipient,
                handle=handle,
                trace=trace,
            )
        )
    return tools


def _build_one_peer_tool(
    *,
    run_id: str,
    sender: str,
    recipient: str,
    handle: Any,
    trace: Callable[[str, dict[str, Any]], None],
) -> BaseTool:
    tool_name = peer_tool_name(recipient)

    async def send_peer_message(
        content: str,
        correlation_id: str | None = None,
        reply_to: str | None = None,
    ) -> dict[str, Any]:
        envelope = build_envelope(
            run_id=run_id,
            sender=sender,
            recipient=recipient,
            content=content,
            correlation_id=correlation_id,
            reply_to=reply_to,
            metadata={'transport': 'academy'},
        )
        await handle.action('receive_message', envelope.model_dump())
        payload = {
            'message_id': envelope.message_id,
            'recipient': recipient,
            'sender': sender,
            'status': 'sent',
        }
        trace('peer_message_sent', payload)
        return payload

    send_peer_message.__name__ = tool_name
    return StructuredTool.from_function(
        coroutine=send_peer_message,
        name=tool_name,
        description=(
            f"Send a graph-to-graph message from {sender} to {recipient}. "
            "Use this only when that peer should act on the message. For one "
            "delegated task, send one complete request and then wait for a "
            "new inbound peer response; do not call this repeatedly to "
            "rephrase the same request, make it canonical, request an "
            "acknowledgement, or poll for status. The returned status 'sent' "
            "only confirms message delivery, not task completion."
        ),
        args_schema=SendPeerMessageInput,
        metadata={
            'chemgraph_tool_kind': 'peer_communication',
            'sender': sender,
            'recipient': recipient,
        },
    )


def peer_tool_name(peer_name: str) -> str:
    return f'send_message_to_{_tool_suffix(peer_name)}'


def _tool_suffix(peer_name: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in peer_name).strip('_')
