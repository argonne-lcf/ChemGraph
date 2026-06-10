"""Adapt Academy actions and campaign FastMCP tools for ChemGraph turns."""

from __future__ import annotations

import json
import pathlib
import time
import uuid
import asyncio
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from academy.handle import Handle
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError

from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.mcp.fastmcp_client import ToolInvocation
from chemgraph.mcp.fastmcp_client import fastmcp_tool_schemas
from chemgraph.mcp.fastmcp_client import (
    FastMCPToolInvoker,
)
from chemgraph.academy.core.peer_protocol import build_message
from chemgraph.academy.observability.run_files import append_jsonl


TraceFn = Callable[[str, dict[str, Any]], None]
SetFinalResultFn = Callable[[dict[str, Any]], None]


@dataclass
class ReasoningToolRuntimeState:
    """Mutable per-turn state updated by ChemGraph reasoning tools."""

    science_tool_completed: bool = False
    submitted_result: bool = False
    finished_turn: bool = False
    executed_tool_names: list[str] = field(default_factory=list)
    action_tool_names: list[str] = field(default_factory=list)
    science_tool_names: list[str] = field(default_factory=list)
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    @property
    def tool_completed(self) -> bool:
        """Backward-compatible name for a completed science tool call."""
        return self.science_tool_completed

    def reset(self) -> None:
        self.science_tool_completed = False
        self.submitted_result = False
        self.finished_turn = False
        self.executed_tool_names.clear()
        self.action_tool_names.clear()
        self.science_tool_names.clear()

    def record_action(self, name: str) -> None:
        self.executed_tool_names.append(name)
        self.action_tool_names.append(name)

    def record_science(self, name: str) -> None:
        self.executed_tool_names.append(name)
        self.science_tool_names.append(name)


class SendMessageArgs(BaseModel):
    """Arguments for the LM-visible peer-message action."""

    model_config = ConfigDict(extra="forbid")

    recipient: str = Field(
        min_length=1,
        description="Allowed peer agent name that should receive this message.",
    )
    tldr: str = Field(
        min_length=1,
        max_length=160,
        description="One-line user-visible summary for dashboard edge labels.",
    )
    content: str = Field(
        min_length=1,
        max_length=1800,
        description="Full peer message content with concise evidence summaries.",
    )
    artifact_refs: list[str] = Field(
        default_factory=list,
        description="JSON array of artifact path strings cited by this message.",
    )
    tool_result_ids: list[str] = Field(
        default_factory=list,
        description="JSON array of ChemGraph tool_result_id strings cited by this message.",
    )
    reply_requested: bool = Field(
        default=False,
        description=(
            "Set true when this message asks the peer to reply or take a "
            "specific follow-up action; false for one-way updates."
        ),
    )
    reason: str = Field(
        min_length=1,
        max_length=600,
        description="Non-empty sentence explaining why this peer needs the message now.",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Numeric confidence from 0 to 1.",
    )


class SubmitResultArgs(BaseModel):
    """Arguments for submitting a logical agent's current result."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1200)
    artifact_refs: list[str] = Field(default_factory=list)
    tool_result_ids: list[str] = Field(default_factory=list)
    supporting_message_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=600)


class FinishTurnArgs(BaseModel):
    """Arguments for ending the current logical-agent turn."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=600)


def _stable_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    """Project Pydantic validation errors to a stable LM-facing shape."""
    return [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": str(error.get("msg", "invalid value")),
        }
        for error in exc.errors()
    ]


def _invalid_args_response(
    tool_name: str,
    exc: ValidationError,
    trace: TraceFn,
) -> dict[str, Any]:
    payload = {
        "tool_name": tool_name,
        "status": "failed",
        "error": "invalid_tool_arguments",
        "error_type": "invalid_tool_arguments",
        "errors": _stable_validation_errors(exc),
    }
    trace("tool_call_failed", payload)
    return {**payload, "status": "error"}


def _disallowed_recipient_response(
    tool_name: str,
    recipient: str,
    allowed: tuple[str, ...],
    trace: TraceFn,
) -> dict[str, Any]:
    payload = {
        "tool_name": tool_name,
        "status": "failed",
        "error": "disallowed_recipient",
        "error_type": "disallowed_recipient",
        "recipient": recipient,
        "allowed_peers": list(allowed),
    }
    trace("tool_call_failed", payload)
    return {**payload, "status": "error"}


def _compact_for_lm(value: Any, *, max_chars: int = 4000) -> Any:
    """Return a JSON-safe, size-bounded value for tool feedback."""
    try:
        text = json.dumps(value, sort_keys=True)
    except TypeError:
        text = repr(value)
    if len(text) <= max_chars:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return {
        "truncated": True,
        "preview": text[:max_chars],
        "full_result_location": "tool_results.jsonl",
    }


async def build_chemgraph_reasoning_tools(
    *,
    spec: ChemGraphAgentSpec,
    run_dir: pathlib.Path,
    tool_invoker: FastMCPToolInvoker,
    peer_names: tuple[str, ...],
    peer_handles: Mapping[str, Handle[Any]],
    outbox: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    get_round_index: Callable[[], int],
    set_final_result: SetFinalResultFn,
    trace: TraceFn,
    runtime_state: ReasoningToolRuntimeState,
) -> list[BaseTool]:
    """Build explicit tools for one ChemGraph-backed reasoning turn."""

    async def _send_message_impl(
        *,
        recipient: str,
        tldr: str,
        content: str,
        artifact_refs: list[str],
        tool_result_ids: list[str],
        reply_requested: bool,
        reason: str,
        confidence: float,
    ) -> dict[str, Any]:
        if recipient not in peer_names:
            raise ValueError(
                f"{spec.name} tried to message disallowed peer {recipient}",
            )
        kind = "question" if reply_requested else "message"
        message = build_message(
            sender=spec.name,
            recipient=recipient,
            content=content,
            round_index=get_round_index(),
            kind=kind,
            tldr=tldr,
            artifact_refs=artifact_refs,
            tool_result_ids=tool_result_ids,
            reply_requested=reply_requested,
            reason=reason,
            confidence=confidence,
        )
        outbox.append(message)
        append_jsonl(run_dir / "messages.jsonl", message)
        trace("message_sent", message)
        if recipient not in peer_handles:
            raise RuntimeError(f"No Academy handle for allowed peer {recipient}")
        task = asyncio.create_task(
            _deliver_message(
                recipient=recipient,
                message=message,
                handle=peer_handles[recipient],
                trace=trace,
            ),
        )
        runtime_state.background_tasks.add(task)
        task.add_done_callback(runtime_state.background_tasks.discard)
        return {
            "status": "sent",
            "delivery": "queued",
            "message_id": message["message_id"],
            "recipient": recipient,
        }

    async def _deliver_message(
        *,
        recipient: str,
        message: dict[str, Any],
        handle: Handle[Any],
        trace: TraceFn,
    ) -> None:
        try:
            await handle.action("receive_message", message)
        except Exception as exc:  # noqa: BLE001 - preserve async delivery failure.
            trace(
                "message_delivery_failed",
                {
                    "recipient": recipient,
                    "message_id": message["message_id"],
                    "error": repr(exc),
                },
            )
            return
        trace(
            "message_delivered",
            {
                "recipient": recipient,
                "message_id": message["message_id"],
            },
        )

    def _validation_error_handler(tool_name: str) -> Callable[[ValidationError], dict[str, Any]]:
        def handle(exc: ValidationError) -> dict[str, Any]:
            runtime_state.record_action(tool_name)
            return _invalid_args_response(tool_name, exc, trace)

        return handle

    async def send_message(**kwargs: Any) -> dict[str, Any]:
        runtime_state.record_action("send_message")
        try:
            args = SendMessageArgs.model_validate(kwargs)
        except ValidationError as exc:
            return _invalid_args_response("send_message", exc, trace)
        if args.recipient not in peer_names:
            return _disallowed_recipient_response(
                "send_message",
                args.recipient,
                peer_names,
                trace,
            )
        return await _send_message_impl(
            recipient=args.recipient,
            tldr=args.tldr,
            content=args.content,
            artifact_refs=args.artifact_refs,
            tool_result_ids=args.tool_result_ids,
            reply_requested=args.reply_requested,
            reason=args.reason,
            confidence=args.confidence,
        )

    async def submit_result(**kwargs: Any) -> dict[str, Any]:
        runtime_state.record_action("submit_result")
        try:
            args = SubmitResultArgs.model_validate(kwargs)
        except ValidationError as exc:
            return _invalid_args_response("submit_result", exc, trace)
        runtime_state.submitted_result = True
        result = {
            "timestamp": time.time(),
            "round": get_round_index(),
            "hypothesis": args.summary,
            "summary": args.summary,
            "artifact_refs": args.artifact_refs,
            "tool_result_ids": args.tool_result_ids,
            "supporting_message_ids": args.supporting_message_ids,
            "supporting_tool_result_ids": args.tool_result_ids,
            "confidence": args.confidence,
            "reason": args.reason,
        }
        set_final_result(result)
        trace("belief_updated", result)
        return {"status": "submitted", "confidence": result["confidence"]}

    async def finish_turn(**kwargs: Any) -> dict[str, Any]:
        runtime_state.record_action("finish_turn")
        try:
            args = FinishTurnArgs.model_validate(kwargs)
        except ValidationError as exc:
            return _invalid_args_response("finish_turn", exc, trace)
        runtime_state.finished_turn = True
        trace("turn_finished_without_external_action", {"reason": args.reason})
        return {"status": "finished", "reason": args.reason}

    tools: list[BaseTool] = [
        StructuredTool.from_function(
            coroutine=send_message,
            name="send_message",
            description=(
                "Send tool-backed evidence, reasoning, or a request to one "
                "allowed peer. Always provide recipient, tldr, content, "
                "artifact_refs as an array of strings or [], tool_result_ids "
                "as an array of strings or [], reply_requested as true when "
                "the peer should respond, a non-empty reason, and numeric "
                "confidence from 0 to 1."
            ),
            args_schema=SendMessageArgs,
            handle_validation_error=_validation_error_handler("send_message"),
            metadata={"chemgraph_academy_tool_kind": "action_tool"},
        ),
        StructuredTool.from_function(
            coroutine=submit_result,
            name="submit_result",
            description=(
                "Submit this agent's current final answer or report. Cite peer "
                "message IDs and ChemGraph tool result IDs."
            ),
            args_schema=SubmitResultArgs,
            handle_validation_error=_validation_error_handler("submit_result"),
            return_direct=True,
            metadata={"chemgraph_academy_tool_kind": "action_tool"},
        ),
        StructuredTool.from_function(
            coroutine=finish_turn,
            name="finish_turn",
            description=(
                "End this decision turn when no tool, message, or report action "
                "is currently useful."
            ),
            args_schema=FinishTurnArgs,
            handle_validation_error=_validation_error_handler("finish_turn"),
            return_direct=True,
            metadata={"chemgraph_academy_tool_kind": "action_tool"},
        ),
    ]

    fastmcp_schemas = await fastmcp_tool_schemas(list(spec.tools))
    schema_by_name = {
        schema["function"]["name"]: schema["function"]
        for schema in fastmcp_schemas
        if schema.get("type") == "function"
    }

    for tool_spec in spec.tools:
        function_schema = schema_by_name[tool_spec.name]

        async def run_fastmcp_tool(
            __tool_name: str = tool_spec.name,
            **kwargs: Any,
        ) -> dict[str, Any]:
            runtime_state.record_science(__tool_name)
            if __tool_name not in spec.tool_names:
                raise RuntimeError(
                    f"{spec.name} cannot call unavailable tool {__tool_name}",
                )
            tool_result_id = f"tool-{uuid.uuid4()}"
            started = {
                "tool_result_id": tool_result_id,
                "tool_name": __tool_name,
                "arguments": kwargs,
            }
            trace("tool_call_started", started)
            result_record = await tool_invoker.invoke(
                ToolInvocation(
                    tool_name=__tool_name,
                    arguments=kwargs,
                    agent_id=spec.name,
                    role=spec.role,
                    correlation_id=tool_result_id,
                ),
            )
            if result_record.status != "success":
                failure = {
                    **started,
                    "status": "failed",
                    "error": result_record.error
                    or "tool returned non-success status",
                }
                append_jsonl(run_dir / "tool_results.jsonl", failure)
                trace("tool_call_failed", failure)
                raise RuntimeError(f"{__tool_name} failed: {failure['error']}")

            runtime_state.science_tool_completed = True
            record = {
                **started,
                "timestamp": time.time(),
                "agent_name": spec.name,
                "status": "ok",
                "result": result_record.result,
            }
            tool_results.append(record)
            append_jsonl(run_dir / "tool_results.jsonl", record)
            trace("tool_call_finished", record)
            return {
                "status": "ok",
                "tool_result_id": tool_result_id,
                "tool_name": __tool_name,
                "result": _compact_for_lm(result_record.result),
            }

        tools.append(
            StructuredTool.from_function(
                coroutine=run_fastmcp_tool,
                name=tool_spec.name,
                description=function_schema.get("description")
                or tool_spec.description
                or f"Run ChemGraph FastMCP tool {tool_spec.name}.",
                args_schema=function_schema.get("parameters")
                or {"type": "object", "properties": {}},
                metadata={"chemgraph_academy_tool_kind": "science_tool"},
            ),
        )

    return tools
