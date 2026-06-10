"""Run one Academy logical-agent wakeup through ChemGraph LangGraph."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from academy.handle import Handle
from langchain_core.tools import BaseTool

from chemgraph.mcp.fastmcp_client import (
    FastMCPToolInvoker,
)
from chemgraph.academy.core.tools import (
    ReasoningToolRuntimeState,
)
from chemgraph.academy.core.tools import (
    build_chemgraph_reasoning_tools,
)
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import visible_resources_payload
from chemgraph.academy.core.lm import LLMSettings
from chemgraph.academy.core.prompt import PromptProfile
from chemgraph.academy.observability.run_files import read_json_file
from chemgraph.academy.observability.run_files import read_jsonl

TraceFn = Callable[[str, dict[str, Any]], None]
SetFinalResultFn = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ReasoningTurnResult:
    """Summary of one ChemGraph-managed logical-agent reasoning turn."""

    final_text: str
    state: dict[str, Any]
    tool_calls_completed: int
    action_tools_called: tuple[str, ...]
    science_tools_called: tuple[str, ...]
    executed_tool_names: tuple[str, ...]
    requested_finish: bool
    requested_self_wake: bool
    workflow_span_id: str
    thread_id: str


class ChemGraphReasoningRoundEngine:
    """Use ChemGraph single_agent as the per-wakeup reasoning loop."""

    def __init__(
        self,
        *,
        campaign: ChemGraphCampaign,
        spec: ChemGraphAgentSpec,
        llm_settings: LLMSettings,
        prompt_profile: PromptProfile,
        run_dir: Path,
        max_decisions: int,
        tools: list[BaseTool],
        runtime_state: ReasoningToolRuntimeState,
        received_message_history: list[dict[str, Any]],
        outbox: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        get_final_result: Callable[[], dict[str, Any] | None],
        get_round_index: Callable[[], int],
        trace: TraceFn,
        peer_names: tuple[str, ...] = (),
    ) -> None:
        self.campaign = campaign
        self.spec = spec
        self.llm_settings = llm_settings
        self.prompt_profile = prompt_profile
        self.run_dir = run_dir
        self.max_decisions = max_decisions
        self.tools = list(tools)
        self.runtime_state = runtime_state
        self.received_message_history = received_message_history
        self.outbox = outbox
        self.tool_results = tool_results
        self.peer_names = tuple(peer_names)
        self.get_final_result = get_final_result
        self.get_round_index = get_round_index
        self.trace = trace

    @classmethod
    async def create(
        cls,
        *,
        campaign: ChemGraphCampaign,
        spec: ChemGraphAgentSpec,
        llm_settings: LLMSettings,
        prompt_profile: PromptProfile,
        run_dir: Path,
        max_decisions: int,
        tool_invoker: FastMCPToolInvoker,
        peer_names: tuple[str, ...],
        peer_handles: Mapping[str, Handle[Any]],
        received_message_history: list[dict[str, Any]],
        outbox: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        get_final_result: Callable[[], dict[str, Any] | None],
        get_round_index: Callable[[], int],
        set_final_result: SetFinalResultFn,
        trace: TraceFn,
    ) -> "ChemGraphReasoningRoundEngine":
        runtime_state = ReasoningToolRuntimeState()
        tools = await build_chemgraph_reasoning_tools(
            spec=spec,
            run_dir=run_dir,
            tool_invoker=tool_invoker,
            peer_names=peer_names,
            peer_handles=peer_handles,
            outbox=outbox,
            tool_results=tool_results,
            get_round_index=get_round_index,
            set_final_result=set_final_result,
            trace=trace,
            runtime_state=runtime_state,
        )
        return cls(
            campaign=campaign,
            spec=spec,
            llm_settings=llm_settings,
            prompt_profile=prompt_profile,
            run_dir=run_dir,
            max_decisions=max_decisions,
            tools=tools,
            runtime_state=runtime_state,
            received_message_history=received_message_history,
            outbox=outbox,
            tool_results=tool_results,
            peer_names=peer_names,
            get_final_result=get_final_result,
            get_round_index=get_round_index,
            trace=trace,
        )

    async def run_turn(self) -> ReasoningTurnResult:
        """Run one turn-local ChemGraph workflow for the current wakeup."""
        from chemgraph.agent.llm_agent import ChemGraph
        from chemgraph.observability.events import WorkflowEventContext
        from chemgraph.observability.events import emit_workflow_event
        from chemgraph.observability.events import new_span_id
        from chemgraph.observability.events import workflow_event_context

        round_index = self.get_round_index()
        thread_id = f"{self.spec.name}-round-{round_index}"
        workflow_span_id = new_span_id(f"chemgraph-turn-{self.spec.name}")
        parent_span_id = f"academy-round-{self.spec.name}-{round_index}"
        query = self.build_wakeup_query(round_index=round_index)
        log_dir = (
            self.run_dir
            / "chemgraph_turns"
            / f"{self.spec.name}-round-{round_index:04d}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        self.runtime_state.reset()
        self.trace(
            "chemgraph_reasoning_turn_started",
            {
                "round": round_index,
                "thread_id": thread_id,
                "workflow_span_id": workflow_span_id,
                "tool_names": [tool.name for tool in self.tools],
            },
        )
        context = WorkflowEventContext(
            run_id=self.run_dir.name,
            run_dir=str(self.run_dir),
            agent_id=self.spec.name,
            role=self.spec.role,
            parent_span_id=parent_span_id,
            tool_name=None,
        )

        with workflow_event_context(
            jsonl_path=self.run_dir / "events.jsonl",
            context=context,
        ):
            emit_workflow_event(
                "workflow_started",
                {
                    "workflow_type": "single_agent",
                    "workflow_node": "ChemGraphReasoningRoundEngine",
                    "round": round_index,
                    "thread_id": thread_id,
                    "tool_names": [tool.name for tool in self.tools],
                    "log_dir": str(log_dir),
                },
                span_id=workflow_span_id,
                parent_span_id=parent_span_id,
            )
            agent = ChemGraph(
                model_name=self.llm_settings.model,
                workflow_type="single_agent",
                base_url=self.llm_settings.base_url,
                api_key=self.llm_settings.api_key,
                argo_user=self.llm_settings.user,
                system_prompt=self.prompt_profile.system_prompt,
                return_option="state",
                recursion_limit=self.prompt_profile.langchain_recursion_limit,
                tools=self.tools,
                terminal_tool_names=("finish_turn", "submit_result"),
                enable_memory=False,
                log_dir=str(log_dir),
            )
            try:
                state = await agent.run(
                    query,
                    config={"configurable": {"thread_id": thread_id}},
                    workflow_span_id=workflow_span_id,
                )
            except Exception as exc:
                emit_workflow_event(
                    "workflow_finished",
                    {
                        "workflow_type": "single_agent",
                        "workflow_node": "ChemGraphReasoningRoundEngine",
                        "round": round_index,
                        "thread_id": thread_id,
                        "status": "failed",
                        "error": repr(exc),
                        "log_dir": str(log_dir),
                    },
                    span_id=workflow_span_id,
                    parent_span_id=parent_span_id,
                )
                raise
            else:
                state = _ensure_state_dict(state)
                emit_workflow_event(
                    "workflow_finished",
                    {
                        "workflow_type": "single_agent",
                        "workflow_node": "ChemGraphReasoningRoundEngine",
                        "round": round_index,
                        "thread_id": thread_id,
                        "status": "completed",
                        "log_dir": str(log_dir),
                    },
                    span_id=workflow_span_id,
                    parent_span_id=parent_span_id,
                )

        if not self.runtime_state.executed_tool_names:
            raise RuntimeError(
                "ChemGraph reasoning turn returned without calling an "
                "Academy action or science tool; logical agents must call "
                "finish_turn when no external action is useful.",
            )

        result = ReasoningTurnResult(
            final_text=_extract_final_text(state),
            state=state,
            tool_calls_completed=len(self.runtime_state.executed_tool_names),
            action_tools_called=tuple(self.runtime_state.action_tool_names),
            science_tools_called=tuple(self.runtime_state.science_tool_names),
            executed_tool_names=tuple(self.runtime_state.executed_tool_names),
            requested_finish=self.runtime_state.finished_turn,
            requested_self_wake=self.runtime_state.science_tool_completed,
            workflow_span_id=workflow_span_id,
            thread_id=thread_id,
        )
        self.trace(
            "chemgraph_reasoning_turn_finished",
            {
                "round": round_index,
                "thread_id": thread_id,
                "workflow_span_id": workflow_span_id,
                "action_tools_called": list(result.action_tools_called),
                "science_tools_called": list(result.science_tools_called),
                "requested_finish": result.requested_finish,
                "requested_self_wake": result.requested_self_wake,
            },
        )
        return result

    def build_wakeup_query(self, *, round_index: int) -> str:
        """Build the user message for one ChemGraph turn."""
        state = self.build_wakeup_state(round_index=round_index)
        return json.dumps(state, sort_keys=True)

    def build_wakeup_state(self, *, round_index: int) -> dict[str, Any]:
        """Build the exact state visible to the logical agent this turn."""
        limits = self.prompt_profile.state_limits
        return {
            "campaign": self.campaign.run_id,
            "user_task": self.campaign.user_task,
            "agent_name": self.spec.name,
            "role": self.spec.role,
            "mission": self.spec.mission,
            "round": round_index,
            "max_decisions": self.max_decisions,
            "resources": visible_resources_payload(self.campaign, self.spec),
            "allowed_peers": list(self.spec.allowed_peers),
            "peer_status": build_peer_status(
                run_dir=self.run_dir,
                peer_names=self.peer_names,
            ),
            "available_chemgraph_tools": list(self.spec.tool_names),
            "received_messages": (
                self.received_message_history[
                    -limits.received_messages_last_n :
                ]
                if limits.received_messages_last_n
                else []
            ),
            "local_chemgraph_tool_results": (
                self.tool_results[-limits.tool_results_last_n :]
                if limits.tool_results_last_n
                else []
            ),
            "recent_actions": build_recent_actions(
                outbox=self.outbox,
                tool_results=self.tool_results,
                limit=limits.actions_last_n,
            ),
            "current_final_result": self.get_final_result(),
            "required_protocol": self.prompt_profile.protocol_prompt,
        }


def build_peer_status(
    *,
    run_dir: Path,
    peer_names: tuple[str, ...],
    event_scan_limit: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Return compact status snapshots for peers visible to this agent."""
    if not peer_names:
        return {}

    now = time.time()
    peers = set(peer_names)
    status: dict[str, dict[str, Any]] = {
        peer: _status_from_agent_file(run_dir, peer, now=now)
        for peer in peer_names
    }

    for event in read_jsonl(run_dir / "events.jsonl")[-event_scan_limit:]:
        agent_id = event.get("agent_id")
        if agent_id not in peers:
            continue
        kind = str(event.get("event") or "")
        timestamp = _float_or_none(event.get("timestamp"))
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        peer_status = status[str(agent_id)]

        if kind == "round_started":
            peer_status["state"] = "busy"
            peer_status["current_activity"] = {
                "type": "reasoning_round",
                "round": payload.get("round"),
                "started_at": timestamp,
            }
            _set_update_age(peer_status, timestamp, now=now)
        elif kind == "tool_call_started":
            peer_status["state"] = "busy"
            peer_status["current_activity"] = {
                "type": "tool_call",
                "tool_name": payload.get("tool_name"),
                "tool_result_id": payload.get("tool_result_id"),
                "tool_call_id": payload.get("tool_call_id"),
                "started_at": timestamp,
            }
            _set_update_age(peer_status, timestamp, now=now)
        elif kind in {"tool_call_finished", "tool_call_failed"}:
            peer_status["state"] = "busy"
            peer_status["current_activity"] = {
                "type": "reasoning_after_tool",
                "last_tool": payload.get("tool_name"),
                "tool_result_id": payload.get("tool_result_id"),
                "status": payload.get("status"),
                "updated_at": timestamp,
            }
            _set_update_age(peer_status, timestamp, now=now)
        elif kind == "message_sent":
            peer_status["last_outbox_tldr"] = (
                payload.get("tldr") or _preview(payload.get("content"))
            )
            peer_status["last_outbox_message_id"] = payload.get("message_id")
            _set_update_age(peer_status, timestamp, now=now)
        elif kind == "belief_updated":
            peer_status["last_belief"] = _compact_belief(payload)
            _set_update_age(peer_status, timestamp, now=now)
        elif kind in {
            "round_finished",
            "turn_finished_without_external_action",
            "workflow_finished",
        }:
            if kind == "workflow_finished" and payload.get("status") == "failed":
                peer_status["state"] = "error"
            else:
                peer_status["state"] = "idle"
            peer_status["current_activity"] = None
            _set_update_age(peer_status, timestamp, now=now)
        elif kind == "agent_error":
            peer_status["state"] = "error"
            peer_status["last_error"] = payload.get("error")
            peer_status["current_activity"] = None
            _set_update_age(peer_status, timestamp, now=now)
        elif kind == "daemon_stopped":
            peer_status["state"] = "finished"
            peer_status["finished"] = True
            peer_status["current_activity"] = None
            _set_update_age(peer_status, timestamp, now=now)

    return status


def _status_from_agent_file(
    run_dir: Path,
    peer_name: str,
    *,
    now: float,
) -> dict[str, Any]:
    data = read_json_file(
        run_dir / "agent_status" / f"{peer_name}.json",
        default={},
    )
    state = "unknown"
    if data:
        if data.get("last_error"):
            state = "error"
        elif data.get("finished") is True:
            state = "finished"
        else:
            state = "idle"
    timestamp = _float_or_none(data.get("status_updated_at"))
    return {
        "state": state,
        "round": data.get("round"),
        "finished": bool(data.get("finished")) if data else False,
        "last_error": data.get("last_error"),
        "current_activity": data.get("current_activity"),
        "seconds_since_update": _age(timestamp, now=now),
        "last_outbox_tldr": _last_outbox_tldr(data),
        "last_outbox_message_id": _last_outbox_message_id(data),
        "last_belief": _compact_belief(data.get("belief")),
    }


def _last_outbox_tldr(data: Mapping[str, Any]) -> str | None:
    recent = data.get("recent_outbox")
    if not isinstance(recent, list) or not recent:
        return None
    last = recent[-1]
    if not isinstance(last, dict):
        return None
    return last.get("tldr") or _preview(last.get("content"))


def _last_outbox_message_id(data: Mapping[str, Any]) -> str | None:
    recent = data.get("recent_outbox")
    if not isinstance(recent, list) or not recent:
        return None
    last = recent[-1]
    if not isinstance(last, dict):
        return None
    value = last.get("message_id")
    return str(value) if value else None


def _compact_belief(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("summary") or value.get("hypothesis")
    if not summary:
        return None
    return {
        "summary": _preview(summary, max_chars=220),
        "confidence": value.get("confidence"),
    }


def _set_update_age(
    peer_status: dict[str, Any],
    timestamp: float | None,
    *,
    now: float,
) -> None:
    peer_status["seconds_since_update"] = _age(timestamp, now=now)


def _age(timestamp: float | None, *, now: float) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, round(now - timestamp, 3))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_recent_actions(
    *,
    outbox: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Build a compact chronological action history for LM prompt state."""
    if limit <= 0:
        return []

    actions: list[dict[str, Any]] = []
    for message in outbox[-limit:]:
        actions.append(
            {
                "type": "send_message",
                "recipient": message.get("recipient"),
                "reply_requested": bool(message.get("reply_requested")),
                "tldr": message.get("tldr") or _preview(message.get("content")),
                "message_id": message.get("message_id"),
                "timestamp": message.get("timestamp"),
            },
        )

    for result in tool_results[-limit:]:
        actions.append(
            {
                "type": "tool_call",
                "tool_name": result.get("tool_name"),
                "tool_result_id": result.get("tool_result_id"),
                "status": result.get("status"),
                "timestamp": result.get("timestamp"),
            },
        )

    actions.sort(key=lambda item: float(item.get("timestamp") or 0.0))
    return actions[-limit:]


def _preview(value: Any, *, max_chars: int = 160) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def _ensure_state_dict(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return state
    return {"value": state}


def _extract_final_text(state: Mapping[str, Any]) -> str:
    messages = state.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        content = last.get("content")
        return "" if content is None else str(content)
    content = getattr(last, "content", None)
    return "" if content is None else str(content)
