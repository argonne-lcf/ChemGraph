"""Service orchestration over the shared preparation and operation journal."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
import json
import logging
import uuid

from langchain_core.messages import ToolMessage

from chemgraph.runtime import OperationJournal, RunContext, SqliteOperationStore
from chemgraph.runtime.journal import APPROVED, BLOCKED, FINALIZED, INVOKING, RETURNED, decode_result
from chemgraph.runtime.policy import PreparedAction
from chemgraph.mcp.service.files import Files, relative_path
from chemgraph.mcp.service.worker import execute, public_value

logger = logging.getLogger(__name__)


def call_from_record(record):
    return {"id": record["tool_call_id"], "name": record["tool"],
            "args": record["prepared"]["requested_arguments"]}


class Service:
    def __init__(self, config):
        self.config = config
        self.store = SqliteOperationStore(config.database)
        self.files = Files(config)
        tools = config.policy.tools - {"molecule_name_to_smiles"}
        if config.enable_pubchem:
            tools = tools | {"molecule_name_to_smiles"}
        self.policy = replace(config.policy, tools=tools)
        self.locks = defaultdict(asyncio.Lock)
        self.inline_slots = asyncio.Semaphore(config.max_inline_tools)
        self.inline_tasks = set()
        self.ready = False
        self.stopping = False
        self.dispatcher = None

    def journal(self, owner):
        return OperationJournal(self.store.for_owner(owner))

    def context(self, owner, identifier, tool):
        root = (self.files.job_workspace(owner, identifier) if tool == "run_ase"
                else self.files.workspace(owner))
        return RunContext(owner=owner, thread_id=identifier, task_id=identifier,
                          workspace=root, policy=self.policy, journal=self.journal(owner))

    def record_context(self, record):
        return self.context(record["owner"], record["thread_id"], record["tool"])

    async def call(self, owner, name, arguments):
        if not self.ready or self.stopping:
            raise ValueError("Service is not ready.")
        if name not in self.policy.tools:
            raise PermissionError("Tool is not permitted.")
        # Keep accepted operations alive if an HTTP client disconnects.
        return await self.track(self._call(owner, name, arguments))

    async def track(self, operation):
        task = asyncio.create_task(operation)
        self.inline_tasks.add(task)
        def completed(done):
            self.inline_tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Retrieve failures even if the HTTP caller disconnected.
        task.add_done_callback(completed)
        return await asyncio.shield(task)

    def stage_arguments(self, context, name, arguments):
        from pydantic_core import to_jsonable_python

        arguments = to_jsonable_python(arguments)
        if name == "run_ase":
            params = arguments.get("params")
            if not isinstance(params, dict):
                raise ValueError("run_ase requires a params object.")
            source = params.get("input_structure_file")
            if not isinstance(source, str) or not source:
                raise ValueError("Provide an input_structure_file or artifact reference.")
            original = self.files.source(context.owner, source)
            self._structure_format(original)
            destination = context.workspace / "inputs" / original.name
            self.files.stage(context.owner, source, destination)
            params["input_structure_file"] = str(destination.relative_to(context.workspace))
            output = relative_path(params.get("output_results_file", "output.json"))
            (context.workspace / output).parent.mkdir(parents=True, exist_ok=True)
            params["output_results_file"] = output
        elif name in {"file_to_atomsdata", "extract_output_json"}:
            key = "fname" if name == "file_to_atomsdata" else "json_file"
            source = arguments.get(key)
            if not isinstance(source, str):
                raise ValueError(f"Provide {key}.")
            original = self.files.source(context.owner, source)
            if name == "file_to_atomsdata":
                self._structure_format(original)
            if source.startswith("artifact:"):
                destination = context.workspace / ".service-staging" / context.task_id / original.name
                self.files.stage(context.owner, source, destination)
                arguments[key] = str(destination.relative_to(context.workspace))
            else:
                relative_path(source)
        elif name in {"save_atomsdata_to_file", "smiles_to_coordinate_file"}:
            key, default = (("fname", "output.xyz") if name == "save_atomsdata_to_file"
                            else ("output_file", "molecule.xyz"))
            path = self.files.path(context.owner, arguments.get(key, default))
            if path.suffix.lower() not in {".xyz", ".cif", ".json"}:
                raise ValueError("Structure outputs must be .xyz, .cif or .json.")
            path.parent.mkdir(parents=True, exist_ok=True)
        self.files.check_quota(context.owner)
        return arguments

    @staticmethod
    def _structure_format(path):
        if path.suffix.lower() not in {".xyz", ".extxyz", ".cif", ".json", ".traj"}:
            raise ValueError("Unsupported service structure format.")

    async def _call(self, owner, name, arguments):
        async with self.inline_slots, self.locks[owner]:
            if self.stopping:
                raise ValueError("Service is stopping.")
            if name == "run_ase":
                pending = await self.store.for_owner(owner).records(state=APPROVED)
                if sum(r["tool"] == "run_ase" for r in pending) >= self.config.per_user_queued:
                    raise ValueError("Per-user queue limit reached.")
            context = self.context(owner, uuid.uuid4().hex, name)
            arguments = await asyncio.to_thread(self.stage_arguments, context, name, arguments)
            call = {"id": context.task_id, "name": name, "args": arguments}
            record = await context.journal.propose(context, call)
            if record["state"] == BLOCKED:
                return self.summary(record)
            await asyncio.to_thread(self.files.check_quota, owner)
            record = await context.journal.decide(record["id"], digest=record["prepared"]["digest"],
                                                  approve=True, decided_by="service-policy")
            if name == "run_ase":
                self.dispatcher.wake.set()
                return {"status": "submitted", "batch_id": record["id"], "state": APPROVED}
            record = await context.journal.begin(context, call, record)
            if record["state"] == BLOCKED:
                return self.summary(record)
            try:
                report = await asyncio.to_thread(execute, PreparedAction.from_dict(record["prepared"]), context.scope)
                await self.returned(context, call, record, report)
            except Exception:
                await self.collection_failed(context, record["id"])
                raise
            return await self.results(owner, record["id"])

    async def collection_failed(self, context, identifier):
        """Reconcile a failed collection without masking its original exception."""
        logger.exception("job_collection_failed owner=%s job=%s", context.owner, identifier)
        try:
            current = await context.journal.get(identifier)
            if current is not None and current["state"] == INVOKING:
                await context.journal.uncertain(identifier, reason="Service lost the invocation outcome.")
        except Exception:
            # If storage is still unavailable, startup recovery handles INVOKING.
            logger.exception("job_reconciliation_deferred owner=%s job=%s", context.owner, identifier)

    async def returned(self, context, call, record, report):
        value = public_value(report["result"], context.workspace)
        message = ToolMessage(content=json.dumps(value, allow_nan=False),
                              tool_call_id=call["id"], name=call["name"],
                              status="error" if report["error"] else "success")
        record = await context.journal.record_return(record, message, report["registered"])
        await self.finalize(record)

    async def finalize(self, record):
        context = self.record_context(record)
        if record["state"] == RETURNED:
            # Bind outputs before a quota failure can defer publication, so later
            # workspace writes cannot replace this operation's results.
            await context.journal.bind_outputs(context, record)
            manifest = record["returned"]["outputs"]
            await asyncio.to_thread(self.files.check_quota, record["owner"], sum(a["size_bytes"] for a in manifest))
            await context.journal.finalize(context, call_from_record(record), record)
            record = await context.journal.get(record["id"])
        # Idempotent catalog registration also repairs a crash after FINALIZED.
        if record["state"] == FINALIZED:
            await asyncio.to_thread(self.files.publish, record["owner"], record["id"], context.workspace,
                                    record["result"]["artifacts"])

    @staticmethod
    def summary(record):
        result = {"batch_id": record["id"], "tool_name": record["tool"], "status": record["state"],
                  "updated_at": record["updated_at"]}
        if record.get("error"):
            result["error"] = record["error"]
        if record["state"] == FINALIZED:
            result["outcome"] = decode_result(record["result"]["message"]).status
        return result

    async def status(self, owner, identifier):
        record = await self.store.for_owner(owner).get(identifier)
        return self.summary(record) if record else {"error": "Unknown job."}

    async def results(self, owner, identifier):
        record = await self.store.for_owner(owner).get(identifier)
        if record is None:
            return {"error": "Unknown job."}
        if record["state"] == RETURNED:
            async with self.locks[owner]:
                record = await self.store.for_owner(owner).get(identifier)
                await self.finalize(record)
                record = await self.store.for_owner(owner).get(identifier)
        result = self.summary(record)
        if record["state"] == FINALIZED:
            # Repair a catalog write that failed after the journal committed.
            await self.finalize(record)
            message = decode_result(record["result"]["message"])
            result["result"] = json.loads(message.content)
            result["artifacts"] = await asyncio.to_thread(self.files.list, owner, operation_id=identifier)
        return result

    async def list_jobs(self, owner, limit=100, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Use a limit between 1 and 100 and a nonnegative offset.")
        records = await self.store.for_owner(owner).records()
        return [self.summary(record) for record in records[offset:offset + limit]]

    async def put_file(self, owner, path, content):
        if not self.ready or self.stopping:
            raise ValueError("Service is not ready.")
        return await self.track(self._put_file(owner, path, content))

    async def _put_file(self, owner, path, content):
        async with self.inline_slots, self.locks[owner]:
            return await asyncio.to_thread(self.files.ingest, owner, path, [content.encode("utf-8")], text=True)
