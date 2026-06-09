from __future__ import annotations

from chemgraph.academy.observability.event_log import CampaignEvent
from chemgraph.academy.observability.payloads import PAYLOAD_MODELS
from chemgraph.academy.observability.payloads import typed_payload


def _payload_for(event: str) -> dict:
    payloads = {
        "message_sent": {
            "message_id": "msg-1",
            "sender": "agent-a",
            "recipient": "agent-b",
            "kind": "message",
            "content": "content",
        },
        "message_received": {
            "message_id": "msg-1",
            "sender": "agent-a",
            "recipient": "agent-b",
            "kind": "message",
            "content": "content",
        },
        "tool_call_started": {
            "tool_result_id": "tool-1",
            "tool_name": "tool",
            "arguments": {},
        },
        "tool_call_finished": {
            "tool_result_id": "tool-1",
            "tool_name": "tool",
            "arguments": {},
            "status": "ok",
        },
        "tool_call_failed": {
            "tool_result_id": "tool-1",
            "tool_name": "tool",
            "arguments": {},
            "status": "failed",
            "error": "boom",
        },
        "workflow_started": {"workflow_type": "single_agent"},
        "workflow_finished": {"workflow_type": "single_agent", "status": "completed"},
        "llm_decision": {},
        "llm_tool_calls": {},
        "agent_started": {},
        "belief_updated": {},
    }
    return payloads[event]


def test_payload_models_round_trip() -> None:
    for event, model in PAYLOAD_MODELS.items():
        payload = _payload_for(event)
        parsed = model.model_validate(payload)
        reparsed = model.model_validate(parsed.model_dump())
        assert reparsed.model_dump() == parsed.model_dump()


def test_typed_payload_selects_model() -> None:
    event = CampaignEvent(
        event="message_sent",
        payload=_payload_for("message_sent"),
    )

    payload = typed_payload(event)

    assert payload is not None
    assert payload.model_dump()["message_id"] == "msg-1"
