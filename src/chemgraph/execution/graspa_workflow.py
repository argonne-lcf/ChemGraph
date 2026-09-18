"""Persisted gRASPA ensemble collection without model-driven polling."""

import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import uuid

from langchain_core.messages import ToolMessage

from chemgraph.tools.graspa_analysis import (
    normalize_record, read_records, write_json, write_records,
)


@contextmanager
def run_lock(root: Path):
    """Exclude concurrent writers; OS locks are released even after a client crash."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".workflow.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another client is using this gRASPA run directory") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another client is using this gRASPA run directory") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def load_journal(root: Path) -> dict:
    return json.loads((root / "workflow.json").read_text())


def save_journal(root: Path, journal: dict) -> None:
    write_json(root / "workflow.json", journal)


async def gather_limited(function, items, config=None):
    """Bound concurrent calls and finish sibling cancellation before returning."""
    concurrency = (config or {}).get("max_concurrency")
    if concurrency is None:
        concurrency = 20
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("max_concurrency must be a positive integer")
    semaphore = asyncio.Semaphore(concurrency)

    async def invoke(item):
        async with semaphore:
            return await function(item)

    tasks = [asyncio.create_task(invoke(item)) for item in items]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def call_tool(tools: dict, name: str, args: dict, config=None):
    """Preserve MCP artifacts by requesting a ToolMessage from the LC adapter."""
    if name not in tools:
        raise ValueError(f"Missing MCP tool: {name}")
    response = await tools[name].ainvoke(
        {"name": name, "args": args, "id": uuid.uuid4().hex, "type": "tool_call"},
        config=config,
    )
    if isinstance(response, ToolMessage):
        if response.status == "error":
            raise RuntimeError(f"{name}: {str(response.content)[:2000]}")
        artifact = response.artifact
        response = (artifact["structured_content"]
                    if isinstance(artifact, dict) and artifact.get("structured_content") is not None
                    else response.content)
    if isinstance(response, list):
        response = "\n".join(
            block["text"] for block in response
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            pass
    if isinstance(response, dict) and set(response) == {"result"}:
        response = response["result"]
    if isinstance(response, dict) and "error" in response:
        raise RuntimeError(f"{name}: {str(response['error'])[:2000]}")
    return response


def validate_records(rows, request: dict, expected_count=None, *, terminal=True, legacy=False):
    if not isinstance(rows, list):
        raise ValueError("MCP results must contain a record list")
    records = [normalize_record(row) for row in rows]
    ids = [r.get("job_id") for r in records if r.get("job_id")]
    if len(ids) != len(set(ids)) or (not legacy and len(ids) != len(records)):
        raise ValueError("Results have missing or duplicate job IDs")
    if expected_count is not None:
        if len(records) > expected_count or (terminal and len(records) != expected_count):
            raise ValueError("Returned record count differs from the submitted task count")
    conditions = Counter((c["temperature"], c["pressure"]) for c in request["conditions"])
    actual = Counter((r["input_structure_file"], r["temperature_in_K"], r["pressure_in_Pa"]) for r in records)
    if any((t, p) not in conditions for _, t, p in actual):
        raise ValueError("Results contain unrequested or missing conditions")
    if isinstance(request.get("input_structures"), list):
        expected = Counter((source, t, p) for source in request["input_structures"]
                           for t, p in conditions.elements())
        if actual - expected or (terminal and actual != expected):
            raise ValueError("Results do not match the requested structures and condition multiplicities")
    elif terminal:
        # Remote discovery determines the source list. Every discovered source
        # must have the full condition multiset, including intentional repeats.
        groups = defaultdict(Counter)
        for (source, t, p), count in actual.items():
            groups[source][t, p] = count
        for counts in groups.values():
            repetitions = [counts[c] / count for c, count in conditions.items()]
            if not repetitions or repetitions[0] < 1 or not repetitions[0].is_integer() or len(set(repetitions)) != 1:
                raise ValueError("Remote results are missing requested condition/repeat pairs")
    if terminal and not records:
        raise ValueError("An empty result set cannot complete an ensemble")
    return records


class GraspaCollector:
    """Collect a frozen workflow while holding its client-side journal lock."""

    def __init__(self, root, tools, options, config=None):
        self.root = Path(root)
        self.tools = {tool.name: tool for tool in tools}
        self.options, self.config = options, config

    async def collect_all(self) -> dict:
        with run_lock(self.root):
            self.journal = load_journal(self.root)

            async def collect(item):
                task_id, request = item
                return task_id, await self.collect(task_id, request)

            return dict(await gather_limited(collect, self.journal["requests"].items(), self.config))

    def save(self):
        save_journal(self.root, self.journal)

    def summary(self, task_id, entry):
        return {"task_id": task_id, "batch_id": entry.get("batch_id"),
                "status": entry.get("status", "collection_error"),
                "total_records": entry.get("total_records", 0),
                "failed_records": entry.get("failed_records", 0),
                "records_path": entry.get("records_path"),
                "message": str(entry.get("message", ""))[:1000]}

    async def collect(self, task_id, request):
        entry = self.journal["batches"].get(task_id)
        if entry is None:
            entry = self.journal["batches"][task_id] = {"phase": "unsubmitted"}
        try:
            async with asyncio.timeout(self.options.wait_timeout_seconds):
                if entry["phase"] in {"submitting", "submission_unknown"}:
                    raise RuntimeError("Submission response is unknown; reconcile MCP list_jobs before resubmitting")
                if entry["phase"] == "collected":
                    rows = read_records(entry["records_path"])
                    self.finish(task_id, entry, request, rows)
                    return self.summary(task_id, entry)
                if entry["phase"] == "unsubmitted":
                    entry.update(phase="submitting", status="submission_unknown")
                    self.save()  # Durable intent before the network call.
                    result = await call_tool(self.tools, "run_graspa_ensemble", {"params": request}, self.config)
                    if isinstance(result, str):
                        match = re.fullmatch(
                            r"Ensemble execution completed\. Ran (\d+) tasks \(\d+ successful\)\. "
                            r"Detailed results saved to '(.*)'\.", result, flags=re.DOTALL,
                        )
                        if not match:
                            raise ValueError("Unrecognized legacy ensemble summary")
                        entry.update(phase="legacy_results", legacy=True,
                                     n_tasks=int(match[1]), legacy_path=match[2])
                        self.save()
                    elif isinstance(result, dict) and result.get("status") == "submitted":
                        # Retain the accepted ID even if a later validation fails.
                        entry.update(phase="submitted", batch_id=result.get("batch_id"), n_tasks=result.get("n_tasks"))
                        self.save()
                    elif isinstance(result, dict) and result.get("status") == "completed":
                        self.finish(task_id, entry, request, result.get("results"))
                        return self.summary(task_id, entry)
                    else:
                        raise ValueError("Unrecognized ensemble response")
                if entry["phase"] == "legacy_results":
                    try:
                        rows = read_records(entry["legacy_path"])
                    except OSError as exc:
                        raise ValueError("Legacy JSONL summary must be readable on the client; use graspa_mcp_hpc for remote workers") from exc
                    self.finish(task_id, entry, request, rows)
                    return self.summary(task_id, entry)
                batch_id, count = entry.get("batch_id"), entry.get("n_tasks")
                if not isinstance(batch_id, str) or not batch_id or type(count) is not int or count <= 0:
                    raise ValueError("Submitted response lacks a valid batch ID or positive n_tasks")
                while True:
                    status = await call_tool(self.tools, "check_job_status", {"batch_id": batch_id}, self.config)
                    if not isinstance(status, dict) or status.get("status") not in {"pending", "running", "completed", "partial", "failed"}:
                        raise ValueError("Invalid batch status response")
                    if status.get("total_tasks") != count:
                        raise ValueError("Batch task count changed after submission")
                    terminal = status["status"] in {"completed", "partial", "failed"}
                    entry.update(status=status["status"], progress=status)
                    result = await call_tool(self.tools, "get_job_results",
                                             {"batch_id": batch_id, "include_partial": True}, self.config)
                    if not isinstance(result, dict):
                        raise ValueError("Invalid batch result response")
                    rows = validate_records(result.get("results"), request, count, terminal=terminal)
                    if terminal:
                        self.finish(task_id, entry, request, rows)
                        return self.summary(task_id, entry)
                    path = self.root / f"{task_id}.jsonl"
                    write_records(path, rows)
                    entry.update(records_path=str(path), total_records=len(rows))
                    self.save()
                    await asyncio.sleep(self.options.poll_interval_seconds)
        except asyncio.CancelledError:
            entry["message"] = "Client collection cancelled; remote work may still be running"
            if entry["phase"] == "submitting":
                entry["phase"] = "submission_unknown"
            self.save()
            raise
        except Exception as exc:
            if entry["phase"] == "submitting":
                entry["phase"] = "submission_unknown"
            entry.update(status="collection_error", message=f"{type(exc).__name__}: {exc}")
            self.save()
            return self.summary(task_id, entry)

    def finish(self, task_id, entry, request, rows):
        records = validate_records(rows, request, entry.get("n_tasks"), legacy=entry.get("legacy", False))
        path = self.root / f"{task_id}.jsonl"
        write_records(path, records)
        failed = sum(r["status"] != "success" for r in records)
        entry.update(phase="collected", total_records=len(records), failed_records=failed,
                     records_path=str(path), message="",
                     status="failed" if failed == len(records) else "partial" if failed else "completed")
        self.save()
