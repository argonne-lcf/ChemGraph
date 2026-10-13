"""Disk-backed reference controller for embedding contract tests.

Production authorization and reconciliation are owned by the embedding service.
This controller deliberately requires an explicit decision for strict contexts.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from chemgraph.execution.prepared import digest, prepare_action
from chemgraph.execution.scoped import EXECUTION_CONTRACT_VERSION, ScopedExecutionError


class DiskExecutionController:
    contract_version = EXECUTION_CONTRACT_VERSION

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()

    def key(self, context, call):
        return digest([context.owner, context.thread_id, context.task_id, call["id"]])

    def get(self, key):
        path = self.directory / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, record):
        path = self.directory / f"{record['id']}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(record, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name == "posix":
            fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def prepare(self, context, call):
        key = self.key(context, call)
        previous = self.get(key)
        if previous and previous["state"] in {"invoking", "returned", "finalized"}:
            raise ScopedExecutionError("An invoked operation cannot be prepared again")
        action = prepare_action(context, call)
        self.put({
            "id": key, "workspace": str(context.workspace), "call_digest": digest(call),
            "state": "prepared", "prepared": action.to_dict(),
        })
        return action

    def decide(self, action, *, approve, expires_in=600):
        record = self.get(action.operation_id)
        if not record or record["state"] not in {"prepared", "nonexecution", "rejected"}:
            raise ValueError("Operation is not awaiting a decision")
        if record["prepared"]["digest"] != action.digest:
            raise ValueError("Decision does not match the prepared digest")
        record.update(state="approved" if approve else "rejected", expires=time.time() + expires_in)
        self.put(record)

    def _workspace_blocker(self, context, key):
        for path in self.directory.glob("*.json"):
            record = json.loads(path.read_text())
            if (record["id"] != key and record["workspace"] == str(context.workspace)
                    and record["state"] in {"invoking", "returned"}):
                return record
        return None

    async def execute_tool(self, context, call, stages):
        async with self.lock:
            key = self.key(context, call)
            record = self.get(key)
            if record and record["call_digest"] != digest(call):
                raise ScopedExecutionError("Operation arguments changed")
            if record and record["state"] == "finalized":
                return record["result"]
            if self._workspace_blocker(context, key):
                raise ScopedExecutionError("Workspace has unfinished invocation or artifact finalization")
            if record and record["state"] == "invoking":
                raise ScopedExecutionError("Invocation outcome is uncertain; reconcile before retrying")
            if not record or record["state"] != "returned":
                if context.policy is not None and (
                    not record or record["state"] != "approved" or record["expires"] <= time.time()
                ):
                    raise ScopedExecutionError("An unexpired approval is required")
                record = record or {
                    "id": key, "workspace": str(context.workspace), "call_digest": digest(call),
                }
                try:
                    preflight = await stages.preflight(record.get("prepared"))
                except Exception as exc:
                    record.update(state="nonexecution", error=str(exc))
                    self.put(record)
                    raise
                record.update(state="invoking", preflight=preflight)
                self.put(record)
                # Any exception/cancellation after intent, including result persistence
                # failure, leaves the durable record uncertain and blocks mutations.
                returned = await stages.invoke(preflight)
                record.update(state="returned", returned=returned)
                self.put(record)
            result = await stages.finalize(record["preflight"], record["returned"])
            record.update(state="finalized", result=result)
            self.put(record)
            return result
