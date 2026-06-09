from __future__ import annotations

from typing import Any

from chemgraph.academy.observability.event_log import CampaignEvent


def build_communication_proof(
    events: list[CampaignEvent],
    placement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build proof that communication could affect recipient behavior."""
    message_ids: dict[str, dict[str, Any]] = {}
    sent_messages: list[dict[str, Any]] = []
    for event in events:
        if event.event != "message_sent":
            continue
        payload = event.payload
        message_id = payload.get("message_id")
        if not isinstance(message_id, str):
            continue
        message = {
            "message_id": message_id,
            "sender": payload.get("sender"),
            "recipient": payload.get("recipient"),
            "content": payload.get("content"),
            "evidence_refs": payload.get("evidence_refs", []),
            "artifact_refs": payload.get("artifact_refs", []),
            "tool_result_ids": payload.get("tool_result_ids", []),
            "timestamp": payload.get("timestamp"),
        }
        message_ids[message_id] = message
        sent_messages.append(message)

    agents = (placement or {}).get("agents", {})
    cross_node_messages = []
    if isinstance(agents, dict):
        for message in sent_messages:
            sender = agents.get(message.get("sender"), {})
            recipient = agents.get(message.get("recipient"), {})
            sender_host = sender.get("short_hostname") or sender.get("hostname")
            recipient_host = recipient.get("short_hostname") or recipient.get("hostname")
            if sender_host and recipient_host and sender_host != recipient_host:
                cross_node_messages.append(
                    {
                        **message,
                        "sender_hostname": sender_host,
                        "recipient_hostname": recipient_host,
                    },
                )

    cited_beliefs = []
    cited_message_ids: set[str] = set()
    for event in events:
        if event.event != "belief_updated":
            continue
        refs = event.payload.get("supporting_message_ids", [])
        if not isinstance(refs, list):
            continue
        cited = [ref for ref in refs if isinstance(ref, str) and ref in message_ids]
        if not cited:
            continue
        cited_message_ids.update(cited)
        cited_beliefs.append(
            {
                "agent_id": event.agent_id,
                "role": event.role,
                "hypothesis": event.payload.get("hypothesis"),
                "confidence": event.payload.get("confidence"),
                "supporting_message_ids": cited,
            },
        )

    cited_tool_refs = []
    final_report_count = 0
    for event in events:
        if event.event != "belief_updated":
            continue
        final_report_count += 1
        refs = (
            event.payload.get("supporting_tool_result_ids")
            or event.payload.get("supporting_artifact_ids")
            or []
        )
        if not isinstance(refs, list):
            continue
        calls = [
            ref
            for ref in refs
            if isinstance(ref, str)
            and (ref.startswith("call-") or ref.startswith("tool-"))
        ]
        if calls:
            cited_tool_refs.append(
                {
                    "agent_id": event.agent_id,
                    "hypothesis": event.payload.get("hypothesis"),
                    "supporting_artifact_ids": calls,
                },
            )

    return {
        "message_count": len(sent_messages),
        "received_message_ids_cited_in_beliefs": sorted(cited_message_ids),
        "belief_changes_citing_messages": len(cited_beliefs),
        "belief_change_examples": cited_beliefs[:10],
        "cross_node_message_count": len(cross_node_messages),
        "cross_node_message_examples": cross_node_messages[:10],
        "tool_refs_cited_in_beliefs": cited_tool_refs[:10],
        "final_report_count": final_report_count,
        "passes": {
            "has_message": bool(sent_messages),
            "has_belief_citing_message": bool(cited_beliefs),
            "has_cross_node_message": bool(cross_node_messages),
            "final_report": final_report_count > 0,
        },
    }
