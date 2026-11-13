"""Read-only integrity reports for scoped records; never replay serialized results."""

from __future__ import annotations

import json
import os
import re
import stat
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

from chemgraph.runtime.artifacts import (
    INPUTS_DIR, artifact_file, checked_path, file_identity, internal_path,
)
from chemgraph.runtime.journal import (
    ABANDONED, APPROVED, BLOCKED, FINALIZED, INVOKING, PROPOSED, REJECTED, RETURNED, UNCERTAIN,
)

_STATES = {ABANDONED, APPROVED, BLOCKED, FINALIZED, INVOKING, PROPOSED, REJECTED, RETURNED, UNCERTAIN}
_STATUSES = ("match", "missing", "mismatch", "invalid_path", "malformed_descriptor", "unreadable")


def _root(value):
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except RuntimeError as exc:  # Symlink loops on Python 3.11/3.12.
        raise ValueError(str(exc)) from exc
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    with os.scandir(root) as entries:
        next(entries, None)  # Surface inaccessible roots, including empty selections.
    return root


def _descriptor(workspace, descriptor, role, index):
    finding = {"role": role, "index": index, "status": "malformed_descriptor"}
    if not isinstance(descriptor, dict):
        return finding
    path, checksum, size = (descriptor.get(key) for key in ("path", "sha256", "size_bytes"))
    if isinstance(path, str):
        finding["path"] = path
    if (not isinstance(path, str) or not path or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum) or type(size) is not int or size < 0):
        return finding
    finding["expected"] = {"sha256": checksum, "size_bytes": size}
    try:
        relative = PurePosixPath(path)
        if (relative.is_absolute() or PureWindowsPath(path).drive
                or ".." in relative.parts or "\\" in path or "\0" in path):
            raise ValueError("Expected a confined POSIX relative path")
        target = (artifact_file(workspace, descriptor) if role == "output" else
                  internal_path(workspace, INPUTS_DIR, relative.relative_to(INPUTS_DIR)))
    except ValueError as exc:
        finding.update(status="invalid_path", detail=str(exc))
        return finding
    except OSError as exc:
        finding.update(status="unreadable", detail=str(exc))
        return finding
    try:
        if not stat.S_ISREG(target.stat().st_mode):
            raise OSError("Not a regular file")
        finding["observed"] = file_identity(target)
        finding["status"] = "match" if finding["observed"] == finding["expected"] else "mismatch"
    except FileNotFoundError:
        finding["status"] = "missing"
    except (OSError, ValueError) as exc:
        finding.update(status="unreadable", detail=str(exc))
    return finding


def _file_entries(journal):
    for path in sorted(journal.iterdir()):
        if not path.name.endswith(".json"):
            continue
        record, issue = None, None
        try:
            checked_path(journal, path)
        except ValueError as exc:
            issue = {"status": "invalid_path", "detail": str(exc)}
        except OSError as exc:
            issue = {"status": "unreadable", "detail": str(exc)}
        if issue is None:
            try:
                if not stat.S_ISREG(path.stat().st_mode):
                    raise OSError("Not a regular journal file")
                record = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, RecursionError) as exc:
                issue = {"status": "malformed_record", "detail": str(exc)}
            except OSError as exc:
                issue = {"status": "unreadable", "detail": str(exc)}
        yield path.name, record, issue


def _record_entries(records):
    for index, record in enumerate(records):
        yield index, record, None


def _audit(entries, workspace, thread_id, include_inputs, journal=None):
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("thread_id must be a nonempty string")
    report = {
        "format_version": 1,
        "selection": {"workspace": str(workspace), "journal": str(journal) if journal else None,
                      "thread_id": thread_id, "include_inputs": include_inputs},
        "selection_complete": True, "records": [], "issues": [],
    }
    counts = Counter(selected=0, finalized=0, skipped=0, excluded=0, requested=0, checked=0)
    statuses = Counter({status: 0 for status in _STATUSES})

    def issue(origin, status, detail):
        report["issues"].append({"origin": origin, "status": status, "detail": detail})
        report["selection_complete"] = False

    entries = iter(entries)
    while True:
        try:
            origin, record, error = next(entries)
        except StopIteration:
            break
        except Exception as exc:  # Only source iteration; audit bugs must still raise.
            issue(None, "unreadable", str(exc))
            break
        if error:
            issue(origin, **error)
            continue
        if (not isinstance(record, dict) or not isinstance(record.get("thread_id"), str)
                or not record["thread_id"]):
            issue(origin, "malformed_record", "Expected an object with a nonempty thread_id")
            continue
        if record["thread_id"] != thread_id:
            counts["excluded"] += 1
            continue
        counts["selected"] += 1
        item = {"origin": origin, "state": record.get("state"), "status": "malformed_record",
                "execution_outcome": "unknown", "scientific_acceptance": "unassessed",
                "descriptors": []}
        item.update({key: record[key] for key in ("id", "task_id", "tool_call_id")
                     if isinstance(record.get(key), str)})
        report["records"].append(item)
        if (not item.get("id") or not isinstance(item["state"], str)
                or item["state"] not in _STATES):
            item["state"] = item["state"] if isinstance(item["state"], str) else None
            issue(origin, "malformed_record", "Expected operation id and known journal state")
            continue
        if item["state"] != FINALIZED:
            item["status"] = "skipped"
            counts["skipped"] += 1
            continue
        counts["finalized"] += 1
        item["status"] = "checked"
        containers = [("output", "result", "artifacts")]
        if include_inputs:
            containers.append(("input", "prepared", "inputs"))
        for role, parent, key in containers:
            container = record.get(parent)
            if not isinstance(container, dict) or not isinstance(container.get(key), list):
                issue(origin, "malformed_record", f"Expected list at {parent}.{key}")
                item["status"] = "issues"
                continue
            for index, descriptor in enumerate(container[key]):
                finding = _descriptor(workspace, descriptor, role, index)
                item["descriptors"].append(finding)
                statuses[finding["status"]] += 1
                counts["requested"] += 1
                counts["checked"] += finding["status"] in {"match", "mismatch"}
                if finding["status"] != "match":
                    item["status"] = "issues"
        if item["status"] == "checked":
            item["status"] = "verified" if item["descriptors"] else "no_descriptors"
    counts["issues"] = len(report["issues"])
    report["counts"] = {**counts, "descriptors": dict(statuses)}
    report["outcome"] = ("issues" if report["issues"] or counts["requested"] != statuses["match"]
                         else "verified" if counts["checked"] else "no_descriptors")
    return report


def audit_records(records, *, workspace, thread_id, include_inputs=False) -> dict:
    """Audit plain records without mutation; source iteration failures become findings.

    Invalid arguments/roots raise ValueError or OSError. Byte integrity does not
    establish execution success, scientific acceptance, or provenance.
    """
    return _audit(_record_entries(records), _root(workspace), thread_id, include_inputs)


def audit_journal(*, workspace, journal, thread_id, include_inputs=False) -> dict:
    """Audit an existing FileOperationStore directory without constructing a store.

    Invalid arguments/roots raise ValueError or OSError. Unreadable/malformed
    entries are reported individually, including entries with unknown threads.
    """
    workspace, journal = _root(workspace), _root(journal)
    return _audit(_file_entries(journal), workspace, thread_id, include_inputs, journal)
