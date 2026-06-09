from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chemgraph.academy.observability.event_log import CampaignEvent


class MessageSentPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    message_id: str
    sender: str
    recipient: str
    content: str
    kind: str | None = None
    tldr: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    tool_result_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    confidence: float | None = None
    round: int | None = None
    timestamp: float | None = None


class MessageReceivedPayload(MessageSentPayload):
    pass


class ToolCallStartedPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool_result_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallFinishedPayload(ToolCallStartedPayload):
    status: str
    result: Any = None
    timestamp: float | None = None
    agent_name: str | None = None


class ToolCallFailedPayload(ToolCallStartedPayload):
    status: str
    error: str


class WorkflowStartedPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    workflow_type: str
    workflow_node: str | None = None
    model_name: str | None = None
    query: str | None = None
    log_dir: str | None = None
    round: int | None = None
    thread_id: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    span_id: str | None = None
    parent_span_id: str | None = None


class WorkflowFinishedPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    workflow_type: str
    status: str
    error: str | None = None
    log_dir: str | None = None
    round: int | None = None
    thread_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None


class LLMDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    round: int | None = None
    tool_names: list[str] = Field(default_factory=list)
    action_tools_called: list[str] = Field(default_factory=list)
    science_tools_called: list[str] = Field(default_factory=list)
    workflow_span_id: str | None = None
    thread_id: str | None = None


class AgentStartedPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    allowed_peers: list[str] = Field(default_factory=list)
    placement: dict[str, Any] | None = None
    hostname: str | None = None
    short_hostname: str | None = None


class BeliefUpdatedPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    hypothesis: str | None = None
    summary: str | None = None
    confidence: float | None = None
    supporting_message_ids: list[str] = Field(default_factory=list)
    supporting_tool_result_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "message_sent": MessageSentPayload,
    "message_received": MessageReceivedPayload,
    "tool_call_started": ToolCallStartedPayload,
    "tool_call_finished": ToolCallFinishedPayload,
    "tool_call_failed": ToolCallFailedPayload,
    "workflow_started": WorkflowStartedPayload,
    "workflow_finished": WorkflowFinishedPayload,
    "llm_decision": LLMDecisionPayload,
    "llm_tool_calls": LLMDecisionPayload,
    "agent_started": AgentStartedPayload,
    "belief_updated": BeliefUpdatedPayload,
}


def typed_payload(event: CampaignEvent) -> BaseModel | None:
    model = PAYLOAD_MODELS.get(event.event)
    return model.model_validate(event.payload) if model else None
