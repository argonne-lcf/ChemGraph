"""Build Academy action tools and attach configured science tools."""

from __future__ import annotations

import pathlib
import time
import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from academy.handle import Handle
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.peer_protocol import build_message
from chemgraph.academy.observability.run_files import append_jsonl


TraceFn = Callable[[str, dict[str, Any]], None]
SetFinalResultFn = Callable[[dict[str, Any]], None]
_BACKGROUND_DELIVERIES: set[asyncio.Task[Any]] = set()


class SendMessageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient: str = Field(min_length=1, description="Allowed peer agent name.")
    tldr: str = Field(min_length=1, max_length=160, description="One-line dashboard edge label.")
    content: str = Field(min_length=1, max_length=1800, description="Full peer message content.")
    artifact_refs: list[str] = Field(default_factory=list, description="Artifact path strings.")
    tool_result_ids: list[str] = Field(default_factory=list, description="ChemGraph tool_result_id strings.")
    reply_requested: bool = Field(
        default=False,
        description="True when this asks the peer to reply or act.",
    )
    reason: str = Field(min_length=1, max_length=600, description="Why this peer needs the message now.")
    confidence: float = Field(ge=0, le=1, description="Numeric confidence from 0 to 1.")
    correlation_id: str | None = Field(
        default=None,
        description="Optional: propagate the correlation_id of the incoming message (workflow/reflex use this to keep replies on the same logical task).",
    )


class SubmitResultArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1200)
    artifact_refs: list[str] = Field(default_factory=list)
    tool_result_ids: list[str] = Field(default_factory=list)
    supporting_message_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=600)


class FinishTurnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=600)


def _stable_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
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


async def build_chemgraph_reasoning_tools(
    *,
    spec: ChemGraphAgentSpec,
    run_dir: pathlib.Path,
    external_tools: Sequence[BaseTool] = (),
    peer_names: tuple[str, ...],
    peer_handles: Mapping[str, Handle[Any]],
    outbox: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    get_round_index: Callable[[], int],
    set_final_result: SetFinalResultFn,
    trace: TraceFn,
    received_message_history: list[dict[str, Any]] | None = None,
    wake_event: asyncio.Event | None = None,
    agent_state: dict[str, dict[str, Any]] | None = None,
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
        correlation_id: str | None = None,
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
            correlation_id=correlation_id,
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
        _BACKGROUND_DELIVERIES.add(task)
        task.add_done_callback(_BACKGROUND_DELIVERIES.discard)
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
            return _invalid_args_response(tool_name, exc, trace)

        return handle

    async def send_message(**kwargs: Any) -> dict[str, Any]:
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
        # Auto-inherit correlation from the last received message when
        # the caller didn't specify one. Keeps replies on the same
        # logical thread without every LLM prompt having to remember.
        effective_corr = args.correlation_id
        if effective_corr is None and received_message_history:
            effective_corr = received_message_history[-1].get("correlation_id")
        return await _send_message_impl(
            recipient=args.recipient,
            tldr=args.tldr,
            content=args.content,
            artifact_refs=args.artifact_refs,
            tool_result_ids=args.tool_result_ids,
            reply_requested=args.reply_requested,
            reason=args.reason,
            confidence=args.confidence,
            correlation_id=effective_corr,
        )

    async def submit_result(**kwargs: Any) -> dict[str, Any]:
        try:
            args = SubmitResultArgs.model_validate(kwargs)
        except ValidationError as exc:
            return _invalid_args_response("submit_result", exc, trace)
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
        try:
            args = FinishTurnArgs.model_validate(kwargs)
        except ValidationError as exc:
            return _invalid_args_response("finish_turn", exc, trace)
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
    tools.extend(external_tools)

    return tools
