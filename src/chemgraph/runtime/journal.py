"""The durable operation state machine for scoped tool calls.

ChemGraph owns the execution states; an embedding service supplies the store
and authenticated decisions. Normal path::

    proposed -> approved -> invoking -> returned -> finalized

Off-path states are ``blocked`` (could not be prepared), ``rejected``,
``uncertain`` (a crash or cancellation after invocation intent was saved) and
``abandoned``. An uncertain operation is never repeated automatically; an
operator either confirms nonexecution (returning it to review) or abandons it.
An exception raised by the tool is an observed outcome and is saved as an
error result. Every save is a compare-and-set on the record's ``revision``, so
a transition never overwrites a concurrent decision.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphBubbleUp

from chemgraph.runtime.artifacts import (
    capture_artifacts, output_manifest, sync_directory, verify_artifacts,
)
from chemgraph.runtime.context import ScopedExecutionError
from chemgraph.runtime.policy import (
    PolicyDenied, PreparationError, PreparedAction, call_digest, operation_id,
    prepare, validate,
)

PROPOSED, BLOCKED, APPROVED, REJECTED = "proposed", "blocked", "approved", "rejected"
INVOKING, RETURNED, FINALIZED = "invoking", "returned", "finalized"
UNCERTAIN, ABANDONED = "uncertain", "abandoned"
UNFINISHED = frozenset({INVOKING, UNCERTAIN, RETURNED})
ARTIFACT_KEY = "chemgraph_artifacts"

_serializer = JsonPlusSerializer()


def encode_result(value) -> dict:
    kind, payload = _serializer.dumps_typed(value)
    return {"type": kind, "data": base64.b64encode(payload).decode("ascii")}


def decode_result(value: dict):
    return _serializer.loads_typed((value["type"], base64.b64decode(value["data"])))


def result_artifacts(state: dict, task_id: str) -> list[dict]:
    """Artifact descriptors of ``task_id`` from live or serialized checkpoint messages."""
    artifacts = {}
    for message in state.get("messages", []):
        extra = (message.get("additional_kwargs", {}) if isinstance(message, dict)
                 else getattr(message, "additional_kwargs", {}))
        for artifact in extra.get(ARTIFACT_KEY, []):
            if artifact.get("task_id") == task_id:
                artifacts[artifact["path"]] = artifact
    return list(artifacts.values())


class OperationConflict(ScopedExecutionError):
    """The stored record changed since it was read; reload it and decide again."""


class ApprovalRequired(ScopedExecutionError):
    """The operation has no valid approval now; it was not invoked."""


class OperationStore(Protocol):
    """Durable operation records. ``put`` must be durable when it returns.

    ``put`` is a compare-and-set: it raises :class:`OperationConflict` unless the
    stored record's ``revision`` equals ``expected_revision`` (``None`` means no
    record with that id exists yet).
    """

    async def get(self, identifier: str) -> dict | None: ...

    async def put(self, record: dict, *, expected_revision: int | None) -> None: ...

    async def records(self, *, thread_id: str | None = None) -> list[dict]: ...


class FileOperationStore:
    """One fsynced JSON file per operation, for single-process embedders and tests."""

    _lock = threading.Lock()

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def get(self, identifier):
        path = self.directory / f"{identifier}.json"
        return json.loads(path.read_text()) if path.exists() else None

    async def put(self, record, *, expected_revision):
        path = self.directory / f"{record['id']}.json"
        temporary = path.with_suffix(".tmp")
        with self._lock:
            current = json.loads(path.read_text())["revision"] if path.exists() else None
            if current != expected_revision:
                raise OperationConflict(f"Operation {record['id']} changed concurrently.")
            with temporary.open("w") as stream:
                json.dump(record, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            sync_directory(self.directory)

    async def records(self, *, thread_id=None):
        records = [json.loads(path.read_text()) for path in sorted(self.directory.glob("*.json"))]
        return [r for r in records if thread_id is None or r["thread_id"] == thread_id]


def default_authorizer(record: dict) -> bool:
    """An unexpired approval of exactly the prepared digest."""
    decision = record.get("decision") or {}
    expires = decision.get("expires_at")
    return (record["state"] == APPROVED and decision.get("approve") is True
            and decision.get("digest") == record["prepared"]["digest"]
            and (expires is None or expires > time.time()))


def error_message(call: dict, content: str) -> ToolMessage:
    return ToolMessage(content=content, name=call["name"], tool_call_id=call["id"], status="error")


# Runs the tool with the prepared arguments, registering outputs into the set.
Invoke = Callable[[PreparedAction, set[str]], Awaitable[ToolMessage]]


class OperationJournal:
    """Review, authorize, execute and reconcile scoped operations."""

    def __init__(self, store: OperationStore, *, authorizer=default_authorizer):
        self.store = store
        self.authorizer = authorizer

    async def get(self, identifier: str) -> dict | None:
        return await self.store.get(identifier)

    async def _put(self, record: dict, state: str, **changes) -> dict:
        expected = record.get("revision")
        record.update(changes, state=state, updated_at=time.time(), revision=(expected or 0) + 1)
        await self.store.put(record, expected_revision=expected)
        return record

    # -- review ---------------------------------------------------------

    async def propose(self, context, call: dict) -> dict:
        """Prepare ``call`` once; later calls return the saved record unchanged."""
        identifier = operation_id(context, call["id"])
        record = await self.store.get(identifier)
        if record is not None:
            if record["call_digest"] != call_digest(call):
                raise ScopedExecutionError(
                    f"Tool call id {call['id']!r} was reused with different arguments."
                )
            return record
        record = {
            "id": identifier, "owner": context.owner, "thread_id": context.thread_id,
            "task_id": context.task_id, "tool_call_id": call["id"], "tool": call["name"],
            "call_digest": call_digest(call), "prepared": None, "decision": None,
            "history": [],
        }
        try:
            action = await asyncio.to_thread(prepare, context, call)
        except (PolicyDenied, PreparationError) as exc:
            state, changes = BLOCKED, {"error": str(exc)}
        else:
            state, changes = PROPOSED, {"prepared": action.to_dict()}
        try:
            return await self._put(record, state, **changes)
        except OperationConflict:
            # A concurrent proposal of this call was saved first.
            return await self.propose(context, call)

    async def decide(self, identifier: str, *, digest: str, approve: bool, decided_by: str,
                     expires_at: float | None = None, message: str | None = None) -> dict:
        """Record an authenticated decision on exactly the prepared digest."""
        record = await self.store.get(identifier)
        if record is None or record["state"] not in {PROPOSED, APPROVED}:
            raise ValueError("The operation is not awaiting a decision.")
        if record["prepared"]["digest"] != digest:
            raise ValueError("The decision does not match the prepared digest.")
        decision = {"approve": approve, "digest": digest, "decided_by": decided_by,
                    "decided_at": time.time(), "expires_at": expires_at, "message": message}
        record["history"].append({"event": "decision", **decision})
        try:
            return await self._put(record, APPROVED if approve else REJECTED, decision=decision)
        except OperationConflict as exc:
            raise ValueError("The operation changed; reload it and decide again.") from exc

    def authorized(self, record: dict) -> bool:
        return self.authorizer(record)

    # -- reconciliation -------------------------------------------------

    async def recover(self) -> list[dict]:
        """Call once at startup: interrupted invocations become uncertain."""
        changed = []
        for record in await self.store.records():
            if record["state"] == INVOKING:
                record["history"].append({"event": "restart", "at": time.time()})
                changed.append(await self._put(record, UNCERTAIN))
        return changed

    async def confirm_nonexecution(self, identifier: str, *, decided_by: str, evidence: str) -> dict:
        """Operator evidence that an uncertain operation did not run; it needs a new approval."""
        record = await self.store.get(identifier)
        if record is None or record["state"] != UNCERTAIN:
            raise ValueError("Only an uncertain operation can be confirmed as not executed.")
        record["history"].append({"event": "nonexecution", "decided_by": decided_by,
                                  "evidence": evidence, "at": time.time()})
        return await self._put(record, PROPOSED, decision=None)

    async def abandon(self, identifier: str, *, decided_by: str, reason: str) -> dict:
        """Stop tracking an operation without claiming success or cancellation."""
        record = await self.store.get(identifier)
        if record is None or record["state"] in {FINALIZED, ABANDONED, REJECTED, BLOCKED}:
            raise ValueError("The operation cannot be abandoned.")
        record["history"].append({"event": "abandoned", "decided_by": decided_by,
                                  "reason": reason, "at": time.time()})
        return await self._put(record, ABANDONED)

    # -- execution ------------------------------------------------------

    async def execute(self, context, call: dict, invoke: Invoke) -> ToolMessage:
        """Run an authorized operation, or replay/finish one already recorded."""
        record = await self.store.get(operation_id(context, call["id"]))
        if record is None or record["call_digest"] != call_digest(call):
            raise ScopedExecutionError("The operation was not reviewed with these arguments.")
        if record["state"] == FINALIZED:
            try:
                await asyncio.to_thread(
                    verify_artifacts, context.workspace, record["result"]["artifacts"])
            except (OSError, ValueError, KeyError) as exc:
                raise ScopedExecutionError(
                    "A saved artifact is missing or corrupt; restore it from backup. "
                    "The calculation will not be repeated."
                ) from exc
            return decode_result(record["result"]["message"])
        if record["state"] == ABANDONED:
            return error_message(call, "An operator abandoned this operation; its outcome is "
                                       "unknown and it will not be repeated.")
        others = [r for r in await self.store.records(thread_id=context.thread_id)
                  if r["id"] != record["id"] and r["state"] in UNFINISHED]
        if others:
            raise ScopedExecutionError(
                "The workspace has an uncertain or unfinished operation; reconcile it first."
            )
        if record["state"] in {INVOKING, UNCERTAIN}:
            raise ScopedExecutionError(
                "The invocation outcome is uncertain; an operator must reconcile it."
            )
        if record["state"] != RETURNED:
            if not self.authorized(record):
                raise ApprovalRequired("The operation has no valid approval.")
            action = PreparedAction.from_dict(record["prepared"])
            try:
                await asyncio.to_thread(validate, context, call, action)
            except (PreparationError, PermissionError, OSError) as exc:
                failure = exc
            else:
                failure = None
            # A decision may have changed or expired meanwhile: transition from
            # the saved record, and only while it is still authorized.
            record = await self.store.get(record["id"])
            if not self.authorized(record):
                raise ApprovalRequired("The approval changed or expired before invocation.")
            changes = {"error": str(failure)} if failure else {}
            try:
                await self._put(record, BLOCKED if failure else INVOKING, **changes)
            except OperationConflict as exc:
                raise ApprovalRequired("The operation changed before invocation.") from exc
            if failure:
                return error_message(call, f"Not executed: {failure} Propose the action again.")
            outputs = set()
            try:
                message = await invoke(action, outputs)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                # The tool raised: its failure is known, unlike a crash.
                message = error_message(call, f"Error: {exc!r}")
            await self._put(record, RETURNED, returned={
                "message": encode_result(message), "registered": sorted(outputs), "outputs": None,
            })
        returned = record["returned"]
        message = decode_result(returned["message"])
        try:
            # Identify outputs once; a retry publishes exactly the saved manifest.
            if returned["outputs"] is None:
                manifest = await asyncio.to_thread(
                    output_manifest, context.workspace, returned["registered"])
                await self._put(record, RETURNED, returned={**returned, "outputs": manifest})
            artifacts = await asyncio.to_thread(
                capture_artifacts, context, call["id"], record["returned"]["outputs"])
        except (OSError, ValueError) as exc:
            raise ScopedExecutionError(
                "Saving the operation's artifacts failed; retry after fixing storage. "
                "The calculation will not be repeated."
            ) from exc
        message = message.model_copy(update={"additional_kwargs": {
            **message.additional_kwargs, ARTIFACT_KEY: artifacts,
        }})
        await self._put(record, FINALIZED,
                        result={"message": encode_result(message), "artifacts": artifacts})
        return message
