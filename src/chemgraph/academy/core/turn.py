"""Run one Academy logical-agent wakeup through ChemGraph."""

from __future__ import annotations
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from langchain_core.tools import BaseTool
from chemgraph.academy.core.campaign import ChemGraphAgentSpec, ChemGraphCampaign
from chemgraph.academy.core.campaign import visible_resources_payload
from chemgraph.academy.core.lm import LLMSettings
from chemgraph.academy.core.prompt import PromptProfile
from chemgraph.academy.observability.run_files import read_json_file
from chemgraph.agent.llm_agent import run_turn

TraceFn = Callable[[str, dict[str, Any]], None]
ACTION_TOOL_NAMES = frozenset({"send_message", "ask_peer", "submit_result", "finish_turn"})
TERMINAL_TOOL_NAMES = ("finish_turn", "submit_result")

@dataclass(frozen=True)
class ReasoningTurnResult:
    final_text: str
    executed_tool_names: tuple[str, ...]
    action_tools_called: tuple[str, ...]
    science_tools_called: tuple[str, ...]
    requested_finish: bool
    requested_self_wake: bool
    thread_id: str

async def run_academy_turn(
    *,
    campaign: ChemGraphCampaign,
    spec: ChemGraphAgentSpec,
    llm_settings: LLMSettings,
    prompt_profile: PromptProfile,
    run_dir: Path,
    max_decisions: int,
    tools: list[BaseTool],
    received_message_history: list[dict[str, Any]],
    outbox: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    get_final_result: Callable[[], dict[str, Any] | None],
    get_round_index: Callable[[], int],
    trace: TraceFn,
    peer_names: tuple[str, ...] = (),
) -> ReasoningTurnResult:
    round_index = get_round_index()
    thread_id = f"{spec.name}-round-{round_index}"
    trace("chemgraph_reasoning_turn_started", {"round": round_index, "thread_id": thread_id, "tool_names": [t.name for t in tools]})

    def on_event(event: str, payload: dict) -> None:
        trace(event, {"round": round_index, **payload})

    result = await run_turn(
        query=json.dumps(_state(campaign, spec, prompt_profile, run_dir, max_decisions, round_index, received_message_history, outbox, tool_results, get_final_result, peer_names), sort_keys=True),
        tools=tools,
        model_name=llm_settings.model,
        base_url=llm_settings.base_url,
        api_key=llm_settings.api_key,
        argo_user=llm_settings.user,
        system_prompt=prompt_profile.system_prompt,
        recursion_limit=prompt_profile.langchain_recursion_limit,
        thread_id=thread_id,
        terminal_tool_names=TERMINAL_TOOL_NAMES,
        on_event=on_event,
    )
    if not result.executed_tool_names:
        raise RuntimeError("ChemGraph reasoning turn returned without calling an Academy action or science tool; call finish_turn when no external action is useful.")
    action_tools = tuple(n for n in result.executed_tool_names if n in ACTION_TOOL_NAMES)
    science_tools = tuple(n for n in result.executed_tool_names if n not in ACTION_TOOL_NAMES)
    out = ReasoningTurnResult(
        final_text=result.final_text,
        executed_tool_names=result.executed_tool_names,
        action_tools_called=action_tools,
        science_tools_called=science_tools,
        requested_finish=result.terminal_tool in TERMINAL_TOOL_NAMES,
        requested_self_wake=bool(science_tools),
        thread_id=result.thread_id,
    )
    trace("chemgraph_reasoning_turn_finished", {"round": round_index, "thread_id": out.thread_id, "action_tools_called": list(action_tools), "science_tools_called": list(science_tools), "requested_finish": out.requested_finish, "requested_self_wake": out.requested_self_wake})
    return out

def _state(campaign, spec, profile, run_dir, max_decisions, round_index, messages, outbox, results, get_final_result, peer_names) -> dict[str, Any]:
    limits = profile.state_limits
    return {
        "campaign": campaign.run_id,
        "user_task": campaign.user_task,
        "agent_name": spec.name,
        "role": spec.role,
        "mission": spec.mission,
        "round": round_index,
        "max_decisions": max_decisions,
        "resources": visible_resources_payload(campaign, spec),
        "allowed_peers": list(spec.allowed_peers),
        "peer_status": build_peer_status(run_dir=run_dir, peer_names=peer_names),
        "available_chemgraph_tools": list(spec.tool_names),
        "received_messages": _tail(messages, limits.received_messages_last_n),
        "local_chemgraph_tool_results": _tail(results, limits.tool_results_last_n),
        "recent_actions": build_recent_actions(outbox=outbox, tool_results=results, limit=limits.actions_last_n),
        "current_final_result": get_final_result(),
        "required_protocol": profile.protocol_prompt,
    }

def build_peer_status(*, run_dir: Path, peer_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {peer: _status(run_dir, peer, now=time.time()) for peer in peer_names}

def build_recent_actions(*, outbox: list[dict[str, Any]], tool_results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    actions = [{"type": "send_message", "recipient": m.get("recipient"), "reply_requested": bool(m.get("reply_requested")), "tldr": m.get("tldr") or _preview(m.get("content")), "message_id": m.get("message_id"), "timestamp": m.get("timestamp")} for m in outbox[-limit:]]
    actions += [{"type": "tool_call", "tool_name": r.get("tool_name"), "tool_result_id": r.get("tool_result_id"), "status": r.get("status"), "timestamp": r.get("timestamp")} for r in tool_results[-limit:]]
    return sorted(actions, key=lambda i: float(i.get("timestamp") or 0.0))[-limit:]

def _status(run_dir: Path, peer: str, *, now: float) -> dict[str, Any]:
    data = read_json_file(run_dir / "agent_status" / f"{peer}.json", default={})
    timestamp = _float(data.get("status_updated_at"))
    state = "unknown" if not data else "error" if data.get("last_error") else "finished" if data.get("finished") else "idle"
    return {"state": state, "round": data.get("round"), "finished": bool(data.get("finished")) if data else False, "last_error": data.get("last_error"), "seconds_since_update": None if timestamp is None else max(0.0, round(now - timestamp, 3))}


def _tail(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return items[-limit:] if limit else []


def _float(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _preview(value: Any, *, max_chars: int = 160) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."
