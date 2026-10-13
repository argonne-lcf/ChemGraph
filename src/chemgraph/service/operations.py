"""Write-ahead computational submissions and human authorization.

An unknown acceptance outcome is never treated as permission to resubmit.
Tool-call IDs identify operations within a task; exact arguments identify approval.
"""

import asyncio
import hashlib
import json
import os
import time

from starlette.responses import JSONResponse
from starlette.routing import Route


class MCPBackend:
    """Lazy transport binding; the backend protocol is ChemGraph-owned."""

    def __init__(self, profile):
        self.profile = profile

    def client(self):
        from chemgraph.execution.submissions import MCPJobClient

        token = os.environ[self.profile.token_env] if self.profile.token_env else None
        return MCPJobClient(self.profile.server_url, token)

    async def submit(self, operation):
        return await self.client().submit(operation)

    async def status(self, batch_id):
        return await self.client().status(batch_id)

    async def results(self, batch_id):
        return await self.client().results(batch_id)

    async def cancel(self, batch_id):
        return await self.client().cancel(batch_id)


class Operations:
    def __init__(self, service, backend=None):
        self.service = service
        self.store = service.store
        self.backend = backend or (
            MCPBackend(service.settings.profile.hpc) if service.settings.profile.hpc else None
        )
        self.io_lock = asyncio.Lock()

    async def get(self, identifier):
        row = await self.store.one("SELECT data FROM operations WHERE id=?", (identifier,))
        return json.loads(row["data"]) if row else None

    async def list(self, task_id):
        rows = await self.store.all("SELECT data FROM operations WHERE task_id=?", (task_id,))
        return [json.loads(row["data"]) for row in rows]

    async def put(self, operation):
        await self.store.db.execute(
            "INSERT OR REPLACE INTO operations VALUES(?,?,?)",
            (operation["id"], operation["task_id"], json.dumps(operation)),
        )

    async def prepare(self, record, call_id, calculation):
        profile = self.service.settings.profile
        if profile.hpc is None:
            raise ValueError("No HPC endpoint is configured")
        template = next((c for c in profile.hpc.calculations if c.name == calculation), None)
        if template is None:
            raise ValueError("Calculation must match an operator-configured template")
        action = {
            "tool": profile.hpc.tool,
            "arguments": template.arguments,
            "server": profile.hpc.server_url,
            "endpoint": profile.hpc.endpoint_id,
            "calculator": profile.hpc.calculator,
            "model": profile.hpc.model,
            "profile": profile.identity,
        }
        digest = hashlib.sha256(json.dumps(action, sort_keys=True).encode()).hexdigest()
        identifier = hashlib.sha256(f"{record['id']}:{call_id}".encode()).hexdigest()
        async with self.service.lock:
            existing = await self.get(identifier)
            if existing:
                if existing["digest"] != digest:
                    raise ValueError("An existing operation changed; approval is invalid")
                return existing
            if len(await self.list(record["id"])) >= profile.max_submissions:
                raise ValueError("Submission budget exhausted")
            operation = {
                "kind": "remote",
                "id": identifier,
                "task_id": record["id"],
                "digest": digest,
                **action,
                "state": "approval",
                "interrupt_id": None,
                "approval": None,
                "attempts": 0,
                "next_poll": 0,
            }
            async with self.store.transaction():
                await self.put(operation)
            return operation

    def native_action(self, record, call):
        from chemgraph.execution.prepared import prepare_action

        action = prepare_action(self.service.adapter.context(record), call)
        return action.operation_id, action.digest, {
            "tool": call["name"], "arguments": call["args"],
            "profile": record["profile"], "prepared": action.to_dict(), "call": call,
            "workspace": str(self.service.adapter.context(record).workspace),
        }

    async def register_reviews(self, record, interrupts, state):
        from chemgraph.execution.prepared import digest

        calls = []
        for message in reversed(state.get("messages", [])):
            if message.get("tool_calls"):
                calls = list(message["tool_calls"])
                break
        async with self.service.lock:
            async with self.store.transaction():
                for item in interrupts:
                    payload = item["payload"]
                    if not isinstance(payload, dict) or "action_requests" not in payload:
                        continue
                    for index, action in enumerate(payload["action_requests"]):
                        call = next((c for c in calls if c["name"] == action["name"]
                                     and c["args"] == action["args"]), None)
                        if call is None:
                            raise ValueError("Review does not match a checkpointed tool call")
                        calls.remove(call)
                        identifier = digest([record["owner"], record["thread_id"], record["id"], call["id"]])
                        existing = await self.get(identifier)
                        if existing:
                            if existing["call"] != call:
                                raise ValueError("Reviewed operation changed")
                            action["prepared"] = existing.get("prepared")
                            action["digest"] = existing["digest"]
                            continue
                        if len(await self.list(record["id"])) >= self.service.settings.profile.max_submissions:
                            raise ValueError("Submission budget exhausted")
                        error = None
                        try:
                            identifier, checksum, details = self.native_action(record, call)
                        except (ValueError, PermissionError, FileNotFoundError, ImportError) as exc:
                            # Missing dependent inputs must be proposed in a later model
                            # turn, after the preceding action has produced its output.
                            error = str(exc)
                            checksum = digest(call)
                            details = {"tool": call["name"], "arguments": call["args"],
                                       "profile": record["profile"], "call": call}
                        operation = {
                            "kind": "native", "id": identifier, "task_id": record["id"],
                            "digest": checksum, **details, "state": "blocked" if error else "approval",
                            "interrupt_id": item["id"], "review_index": index, "approval": None,
                        }
                        if error:
                            operation["preparation_error"] = error
                        action["prepared"] = details.get("prepared")
                        action["digest"] = checksum
                        action["preparation_error"] = error
                        await self.put(operation)

    @staticmethod
    def authorized(operation):
        grant = operation.get("approval")
        return bool(
            grant
            and grant["expires"] > time.time()
            and grant["digest"] == operation["digest"]
            and grant["interrupt_id"] == operation["interrupt_id"]
        )

    async def resume_responses(self, record):
        async with self.service.lock:
            operations = await self.list(record["id"])
            pending_ids = {item["id"] for item in record["interrupts"]}
            reviewed = [op for op in operations if op.get("interrupt_id") in pending_ids]
            if not reviewed:
                return record.get("responses") or {
                    item["id"]: record["text"] for item in record["interrupts"]
                }
            if any(op["state"] not in {"rejected", "blocked"} and not self.authorized(op) for op in reviewed):
                return None
            responses = {}
            for item in record["interrupts"]:
                native = sorted(
                    (
                        op
                        for op in reviewed
                        if op["kind"] == "native" and op["interrupt_id"] == item["id"]
                    ),
                    key=lambda op: op["review_index"],
                )
                responses[item["id"]] = (
                    {"decisions": [
                        {"type": "reject", "message": op.get("rejection", {}).get("message")
                         or op.get("preparation_error", "Operator rejected this action")}
                        if op["state"] in {"rejected", "blocked"} else {"type": "approve"}
                        for op in native
                    ]}
                    if native
                    else "operator-authorized"
                )
            # Acceptance is durable before the graph can execute reviewed tools,
            # including checkpoint-backed workspace tools outside the native registry.
            async with self.store.transaction():
                for op in reviewed:
                    if op["kind"] == "native" and op["state"] not in {"rejected", "blocked"}:
                        op["state"] = "authorized"
                        await self.put(op)
            return responses

    async def execute_tool(self, record, context, call, stages):
        from chemgraph.execution.prepared import digest
        from chemgraph.execution.scoped import ScopedExecutionError

        identifier = digest([context.owner, context.thread_id, context.task_id, call["id"]])
        async with self.service.lock:
            operation = await self.get(identifier)
            if not operation or operation["kind"] != "native":
                raise ScopedExecutionError("Native operation requires exact operator review")
            if operation["call"] != call:
                raise ScopedExecutionError("Native operation arguments changed")
            if operation["state"] in {"completed", "rejected"} and operation.get("result"):
                return operation["result"]
            all_operations = await self.store.all("SELECT data FROM operations")
            for row in all_operations:
                other = json.loads(row["data"])
                if (other["id"] != identifier and other.get("workspace") == str(context.workspace)
                        and other["state"] in {"invoking", "uncertain", "returned"}):
                    raise ScopedExecutionError("Workspace has unfinished invocation or finalization")
            if operation["state"] in {"invoking", "uncertain"}:
                raise ScopedExecutionError("Invocation is uncertain; operator reconciliation required")
            if operation["state"] != "returned":
                if operation["state"] != "authorized" or not self.authorized(operation):
                    raise ScopedExecutionError("An unexpired exact approval is required")
                try:
                    preflight = await stages.preflight(operation["prepared"])
                except Exception as exc:
                    operation.update(state="nonexecution", preflight_error=str(exc), approval=None)
                    async with self.store.transaction():
                        await self.put(operation)
                    raise
                operation.update(state="invoking", preflight=preflight)
                async with self.store.transaction():
                    await self.put(operation)
        if operation["state"] != "returned":
            returned = await stages.invoke(operation["preflight"])
            operation.update(state="returned", returned=returned)
            async with self.service.lock:
                async with self.store.transaction():
                    await self.put(operation)
        result = await stages.finalize(operation["preflight"], operation["returned"])
        operation.update(state="completed", result=result)
        async with self.service.lock:
            async with self.store.transaction():
                await self.put(operation)
        return result

    async def execute(self, identifier):
        async with self.service.lock:
            operation = await self.get(identifier)
            if operation["state"] in {"running", "completed", "failed", "canceled", "uncertain", "rejected", "abandoned"}:
                return operation
            grant = operation.get("approval")
            if (
                not grant
                or grant["expires"] <= time.time()
                or grant["digest"] != operation["digest"]
            ):
                return operation
            if operation["state"] == "submitting":
                return {**operation, "state": "uncertain"}
            operation["state"] = "submitting"
            async with self.store.transaction():
                await self.put(operation)
        try:
            receipt = await self.backend.submit(operation)
            if not receipt.get("batch_id"):
                raise ValueError("Submission returned no batch identifier")
        except Exception:
            operation["state"] = "uncertain"
        else:
            operation.update(state="running", receipt=receipt, batch_id=receipt["batch_id"])
        async with self.service.lock:
            async with self.store.transaction():
                await self.put(operation)
        return operation

    async def recover(self):
        async with self.service.lock:
            async with self.store.transaction():
                rows = await self.store.all("SELECT data FROM operations")
                for row in rows:
                    operation = json.loads(row["data"])
                    if operation["state"] in {"submitting", "invoking"}:
                        operation["state"] = "uncertain"
                        await self.put(operation)

    async def bind_interrupts(self, record, interrupts):
        for item in interrupts:
            payload = item["payload"]
            if not isinstance(payload, dict) or "operation_id" not in payload:
                continue
            operation = await self.get(payload["operation_id"])
            if operation and operation["task_id"] == record["id"]:
                operation["interrupt_id"] = item["id"]
                await self.put(operation)

    async def poll(self):
        async with self.io_lock:
            await self._poll()

    async def _poll(self):
        rows = await self.store.all("SELECT data FROM operations")
        for row in rows:
            operation = json.loads(row["data"])
            if operation["state"] != "running" or operation["next_poll"] > time.time():
                continue
            # Do not contact a different endpoint after a configuration change.
            if operation["profile"] != self.service.settings.profile.identity:
                continue
            try:
                status = await self.backend.status(operation["batch_id"])
                operation["last_status"] = status
                operation.pop("availability_error", None)
                if status.get("status") in {"completed", "failed", "partial"}:
                    operation["results"] = await self.backend.results(operation["batch_id"])
                    operation["state"] = (
                        "completed" if status["status"] == "completed" else "failed"
                    )
            except Exception:
                operation["availability_error"] = "Backend unavailable; tracking continues"
            operation["attempts"] += 1
            operation["next_poll"] = time.time() + min(300, 5 * 2 ** min(operation["attempts"], 6))
            async with self.service.lock:
                async with self.store.transaction():
                    await self.put(operation)

    async def cancel(self, task_id):
        async with self.io_lock:
            for operation in await self.list(task_id):
                if operation["state"] != "running":
                    continue
                if operation["profile"] != self.service.settings.profile.identity:
                    continue
                try:
                    status = await self.backend.status(operation["batch_id"])
                    receipt = await self.backend.cancel(operation["batch_id"])
                    operation["cancellation"] = receipt
                    # Missing upstream futures are reported as already_done, not canceled.
                    total = status.get("total_tasks", 0)
                    if total > 0 and receipt.get("cancelled") == total:
                        operation["state"] = "canceled"
                except Exception:
                    operation["cancellation"] = {
                        "confirmed": False,
                        "reason": "backend unavailable",
                    }
                async with self.service.lock:
                    async with self.store.transaction():
                        await self.put(operation)

    async def reconcile(self, identifier, batch_id, evidence, operator, *, mode="job", result=None):
        from chemgraph.execution.scoped import encode_result
        from langchain_core.messages import ToolMessage
        from a2a.types import a2a_pb2 as a2a
        from .protocol import transition

        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Reviewed evidence is required")
        if mode not in {"job", "result", "nonexecution", "abandon", "finalize"}:
            raise ValueError("Unknown reconciliation mode")
        async with self.service.lock:
            operation = await self.get(identifier)
            if not operation or operation["state"] not in {
                "uncertain", "invoking", "returned", "nonexecution", "authorized",
            }:
                raise ValueError("Only unfinished operations can be reconciled")
            record = await self.store.get(operation["task_id"])
            if record["phase"] != "waiting" or record.get("reason") != "reconciliation":
                raise ValueError("Wait for the active worker to stop before reconciliation")
            if record["profile"] != self.service.settings.profile.identity:
                raise ValueError("Restore the original profile first")
            operation["reconciliation"] = {"operator": operator, "evidence": evidence,
                                           "mode": mode, "time": time.time()}
            operation.setdefault("reconciliations", []).append({
                **operation["reconciliation"], "previous_state": operation["state"],
                "digest": operation["digest"],
            })
            if mode == "job":
                if operation["state"] != "uncertain" or operation["kind"] != "remote" or not isinstance(batch_id, str) or not batch_id:
                    raise ValueError("A recovered remote job identifier is required")
                operation.update(state="running", batch_id=batch_id, receipt={"batch_id": batch_id})
                record.update(phase="queued" if record["interrupts"] else "backend",
                              action="resume", text="operator-reconciled")
            elif mode == "abandon":
                operation.update(state="abandoned", approval=None)
                record.update(phase="done", interrupts=[])
                transition(record, a2a.TASK_STATE_FAILED,
                           "Operation abandoned; completion and cancellation are not confirmed.", "abandoned")
            elif mode == "nonexecution":
                if operation["state"] == "returned":
                    raise ValueError("A saved result proves invocation; cannot confirm nonexecution")
                if operation["kind"] == "native":
                    operation.setdefault("previous_preparations", []).append(operation["prepared"])
                    _, checksum, details = self.native_action(record, operation["call"])
                    operation.update(details, digest=checksum)
                operation.update(state="approval", approval=None)
                if operation["kind"] == "native":
                    operation["interrupt_id"] = "renewed-" + hashlib.sha256(
                        f"{identifier}:{time.time_ns()}".encode()).hexdigest()
                    record["interrupts"] = [{"id": operation["interrupt_id"],
                                              "payload": {"kind": "approval", "operation_id": identifier}}]
                    record["resume_action"] = "recover"
                record.update(phase="waiting", reason="approval")
                transition(record, a2a.TASK_STATE_INPUT_REQUIRED,
                           "Confirmed nonexecution; a new exact approval is required.", "approval",
                           interrupts=record["interrupts"])
            else:
                if operation["kind"] != "native" or not operation.get("preflight"):
                    raise ValueError("Native recovery requires a saved output baseline")
                if mode == "result":
                    if operation["state"] not in {"uncertain", "invoking"} or result is None:
                        raise ValueError("Supply a recovered result only when no returned result is saved")
                    message = ToolMessage(content=json.dumps(result), tool_call_id=operation["call"]["id"])
                    operation.update(state="returned", returned={"result": encode_result(message)})
                elif operation["state"] != "returned":
                    raise ValueError("No saved result awaits finalization")
                record.update(phase="queued", action="recover", text="operator-reconciled")
            if mode in {"job", "result", "finalize"}:
                transition(record, a2a.TASK_STATE_WORKING, "Recovered execution is being finalized")
            async with self.store.transaction():
                await self.put(operation)
                await self.store.put(record)
                if mode == "abandon":
                    await self.store.db.execute("DELETE FROM contexts WHERE owner=? AND context_id=?",
                                               (record["owner"], record["context_id"]))
            await self.service.project(record)
            self.service.wake.set()
            return {"operation_id": identifier, "state": operation["state"]}

    async def approve(self, identifier, data, operator, *, reject=False):
        async with self.service.lock:
            operation = await self.get(identifier)
            if not operation or operation["state"] not in ({"approval", "blocked"} if reject else {"approval"}):
                raise ValueError("Operation is not awaiting approval")
            if (
                data.get("digest") != operation["digest"]
                or not operation["interrupt_id"]
                or data.get("interrupt_id") != operation["interrupt_id"]
            ):
                raise ValueError("Approval must match action digest and pending interrupt")
            expires_in = int(data.get("expires_in", 600))
            if not 1 <= expires_in <= 3600:
                raise ValueError("Approval lifetime must be between 1 and 3600 seconds")
            record = await self.store.get(operation["task_id"])
            if record["phase"] != "waiting" or record.get("reason") != "approval":
                raise ValueError("Task is not awaiting approval")
            if record["profile"] != self.service.settings.profile.identity:
                raise ValueError("Execution profile changed")
            operation["approval"] = {
                "operator": operator,
                "digest": operation["digest"],
                "interrupt_id": operation["interrupt_id"],
                "expires": time.time() + expires_in,
                "chat_actor": data.get("chat_actor"),
            }
            if reject:
                operation.update(state="rejected", approval=None,
                                 rejection={"operator": operator, "message": data.get("message", "Operator rejected action"),
                                            "digest": operation["digest"], "time": time.time()})
            operation.setdefault("decisions", []).append({
                "type": "reject" if reject else "approve",
                **(operation["rejection"] if reject else operation["approval"]),
                "time": time.time(),
            })
            if record.get("resume_action") == "recover":
                if reject:
                    from chemgraph.execution.scoped import encode_result
                    from langchain_core.messages import ToolMessage

                    operation["result"] = {"result": encode_result(ToolMessage(
                        content=operation["rejection"]["message"],
                        tool_call_id=operation["call"]["id"], status="error",
                    )), "artifacts": []}
                else:
                    operation["state"] = "authorized"
            async with self.store.transaction():
                await self.put(operation)
                pending_ids = {item["id"] for item in record["interrupts"]}
                pending = [
                    op
                    for op in await self.list(record["id"])
                    if op.get("interrupt_id") in pending_ids
                ]
                if not pending or any(op["state"] not in {"rejected", "blocked"} and not self.authorized(op) for op in pending):
                    return {"operation_id": identifier, "approved_by": operator, "waiting": True}
                record.update(phase="queued", action=record.get("resume_action", "resume"), text="operator-authorized")
                from a2a.types import a2a_pb2 as a2a

                from .protocol import transition

                transition(record, a2a.TASK_STATE_SUBMITTED, "Operator authorization recorded")
                await self.store.put(record)
            await self.service.project(record)
            self.service.wake.set()
            return {"operation_id": identifier, "approved_by": operator}


def operation_routes(service):
    async def inspect(request):
        return JSONResponse(await service.operations.list(request.path_params["task_id"]))

    async def approve(request):
        try:
            result = await service.operations.approve(
                request.path_params["operation_id"], await request.json(), request.user.display_name
            )
        except (ValueError, TypeError):
            return JSONResponse(
                {"error": "Approval does not match the pending action"}, status_code=409
            )
        return JSONResponse(result)

    async def reject(request):
        try:
            result = await service.operations.approve(
                request.path_params["operation_id"], await request.json(), request.user.display_name,
                reject=True,
            )
        except (ValueError, TypeError):
            return JSONResponse({"error": "Rejection does not match the pending action"}, status_code=409)
        return JSONResponse(result)

    async def reconcile(request):
        data = await request.json()
        try:
            await service.operations.reconcile(
                request.path_params["operation_id"],
                data.get("batch_id"),
                data.get("evidence"),
                request.user.display_name,
                mode=data.get("mode", "job"), result=data.get("result"),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"ok": True})

    return [
        Route("/admin/tasks/{task_id}/operations", inspect),
        Route("/admin/operations/{operation_id}/approve", approve, methods=["POST"]),
        Route("/admin/operations/{operation_id}/reject", reject, methods=["POST"]),
        Route("/admin/operations/{operation_id}/reconcile", reconcile, methods=["POST"]),
    ]
