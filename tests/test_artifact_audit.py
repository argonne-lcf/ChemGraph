"""Scoped artifact inspection is read-only and resilient to individual failures."""

import copy
import json
import os
from pathlib import Path

import pytest

from chemgraph.runtime import FileOperationStore, OperationJournal, RunContext
from chemgraph.runtime import artifact_audit as audit
from chemgraph.runtime import journal as journal_module
from chemgraph.runtime.artifacts import file_identity


@pytest.fixture
def sample(tmp_path):
    workspace, journal = tmp_path / "workspace", tmp_path / "journal"
    journal.mkdir()
    descriptors = []
    for directory in (".chemgraph-artifacts", ".chemgraph-inputs"):
        path = workspace / directory / "key" / "sample.xyz"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"sample")
        descriptors.append({"path": path.relative_to(workspace).as_posix(), **file_identity(path)})
    record = {"id": "op", "thread_id": "thread", "state": "finalized",
              "result": {"artifacts": [descriptors[0]], "message": "never decode"},
              "prepared": {"inputs": [descriptors[1]], "resources": [{"path": "/never/open"}]}}
    return workspace, journal, record


def inspect(sample, *, records=None, inputs=False):
    workspace, _, record = sample
    return audit.audit_records([record] if records is None else records,
                               workspace=workspace, thread_id="thread", include_inputs=inputs)


def status(report):
    return report["records"][0]["descriptors"][0]["status"]


def inventory(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}


def test_read_only_api_and_file_reader(sample, monkeypatch):
    workspace, journal, record = sample
    (journal / "op.json").write_text(json.dumps(record))
    (journal / "ignored.tmp").write_text("incomplete")
    before, original = inventory(workspace.parent), copy.deepcopy(record)
    def forbidden(*args, **kwargs):
        pytest.fail("Audit must not initialize, mutate, execute, or decode runtime state")
    for obj, name in ((RunContext, "__post_init__"), (FileOperationStore, "__init__"),
                      (FileOperationStore, "put"), (OperationJournal, "execute"),
                      (OperationJournal, "recover"), (journal_module, "decode_result")):
        monkeypatch.setattr(obj, name, forbidden)
    report = inspect(sample, records=iter([record]), inputs=True)
    disk = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread", include_inputs=True)
    assert report["outcome"] == disk["outcome"] == "verified"
    assert report["counts"]["checked"] == 2
    assert disk["records"][0]["origin"] == "op.json"
    assert report["records"][0]["execution_outcome"] == "unknown"
    assert report["records"][0]["scientific_acceptance"] == "unassessed"
    assert record == original and inventory(workspace.parent) == before


@pytest.mark.parametrize("change,expected", [("delete", "missing"), ("bytes", "mismatch"),
                                            ("size", "mismatch"), ("directory", "unreadable"),
                                            ("empty", "match")])
def test_byte_findings(sample, change, expected):
    workspace, _, record = sample
    descriptor = record["result"]["artifacts"][0]
    path = workspace / descriptor["path"]
    saved_stat = path.stat()
    if change in {"delete", "directory"}:
        path.unlink()
        if change == "directory":
            path.mkdir()
    elif change == "size":
        descriptor["size_bytes"] += 1
    else:
        path.write_bytes(b"" if change == "empty" else b"change")
        os.utime(path, ns=(saved_stat.st_atime_ns, saved_stat.st_mtime_ns))
        if change == "empty":
            descriptor.update(file_identity(path))
    before = inventory(workspace)
    report = inspect(sample)
    assert status(report) == expected
    assert report["selection_complete"]
    assert report["counts"]["checked"] == (expected in {"match", "mismatch"})
    assert inventory(workspace) == before


@pytest.mark.parametrize("key,value", [("path", None), ("path", ""), ("sha256", "g" * 64),
                                     ("sha256", "a" * 63), ("sha256", "A" * 64), ("sha256", None),
                                     ("size_bytes", -1), ("size_bytes", True), ("size_bytes", 1.0)])
def test_malformed_descriptor_preserves_siblings(sample, key, value):
    record = sample[2]
    descriptors = record["result"]["artifacts"]
    bad = {**descriptors[0], key: value}
    descriptors[:0] = [bad, None, {}]
    report = inspect(sample)
    assert [d["status"] for d in report["records"][0]["descriptors"]] == ["malformed_descriptor"] * 3 + ["match"]
    assert report["outcome"] == "issues" and report["counts"]["checked"] == 1


@pytest.mark.parametrize("role", ["output", "input"])
@pytest.mark.parametrize("path", ["/etc/passwd", "C:/outside", "C:outside", "//host/share/file",
                                ".chemgraph-artifacts/../outside", ".chemgraph-inputs/sample",
                                ".chemgraph-artifacts/..\\outside", ".chemgraph-artifacts/\0"])
def test_invalid_paths_never_hash(sample, monkeypatch, path, role):
    if role == "input":
        sample[2]["result"]["artifacts"] = []
        sample[2]["prepared"]["inputs"][0]["path"] = path.replace(".chemgraph-inputs", ".chemgraph-artifacts")
    else:
        sample[2]["result"]["artifacts"][0]["path"] = path
    monkeypatch.setattr(audit, "file_identity", lambda *_: pytest.fail("Invalid path was opened"))
    assert status(inspect(sample, inputs=role == "input")) == "invalid_path"


@pytest.mark.parametrize("kind", ["output", "input", "parent", "reserved", "journal"])
def test_symlinks_rejected(sample, kind):
    workspace, journal, record = sample
    output = workspace / record["result"]["artifacts"][0]["path"]
    target = (journal / "op.json" if kind == "journal" else
              workspace / record["prepared"]["inputs"][0]["path"] if kind == "input" else
              output.parent if kind == "parent" else output.parent.parent if kind == "reserved" else output)
    if kind == "journal":
        target.write_text(json.dumps(record))
    moved = target.with_name(target.name + "-original")
    target.rename(moved)
    try:
        target.symlink_to(moved, target_is_directory=moved.is_dir())
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    if kind == "journal":
        report = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread")
        assert not report["selection_complete"] and report["issues"][0]["status"] == "invalid_path"
    else:
        report = inspect(sample, inputs=True)
        assert report["records"][0]["descriptors"][kind == "input"]["status"] == "invalid_path"


@pytest.mark.parametrize("bad", [b"{", b"\xff", b"[]", b"{}", b"[" * 2000 + b"]" * 2000])
def test_bad_journal_entry_does_not_hide_valid_records(sample, bad):
    workspace, journal, record = sample
    (journal / "a.json").write_bytes(bad)
    (journal / "b.json").write_text(json.dumps(record))
    report = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread")
    assert report["outcome"] == "issues" and not report["selection_complete"]
    assert report["counts"]["checked"] == 1 and report["issues"][0]["origin"] == "a.json"


@pytest.mark.parametrize("parent,key", [("result", "artifacts"), ("prepared", "inputs")])
@pytest.mark.parametrize("value", ["missing", None, {}, "invalid"])
def test_malformed_container_checks_other_role(sample, parent, key, value):
    if value == "missing":
        del sample[2][parent][key]
    else:
        sample[2][parent][key] = value
    report = inspect(sample, inputs=True)
    assert not report["selection_complete"] and report["counts"]["checked"] == 1
    assert report["outcome"] == "issues"


def test_selection_empty_skipped_and_malformed_states(sample):
    record = sample[2]
    skipped = [{**record, "state": state, "result": None} for state in
               ("proposed", "approved", "blocked", "rejected", "invoking", "returned", "uncertain", "abandoned")]
    excluded = {"thread_id": "other", "result": "do not inspect"}
    report = inspect(sample, records=[excluded, *skipped, record])
    assert report["outcome"] == "verified" and report["counts"]["excluded"] == 1
    assert report["counts"]["skipped"] == len(skipped)
    for records in ([], [excluded], skipped, [{**record, "result": {"artifacts": []}}]):
        assert inspect(sample, records=records)["outcome"] == "no_descriptors"
    for state in (None, "future", []):
        assert inspect(sample, records=[{**record, "state": state}])["outcome"] == "issues"


def test_input_scope_and_iterator_failure(sample):
    workspace, _, record = sample
    (workspace / record["prepared"]["inputs"][0]["path"]).unlink()
    record["result"]["artifacts"][0]["source_path"] = "/never/open"
    record["prepared"]["inputs"][0]["source"] = "/never/open"
    assert inspect(sample)["outcome"] == "verified"
    assert inspect(sample, inputs=True)["outcome"] == "issues"
    def broken():
        yield record
        raise OSError("custom store unavailable")
    report = inspect(sample, records=broken())
    assert report["outcome"] == "issues" and report["counts"]["checked"] == 1
    assert not report["selection_complete"]
    class Unavailable:
        def __iter__(self):
            raise OSError("cannot start iterator")
    assert inspect(sample, records=Unavailable())["outcome"] == "issues"


@pytest.mark.parametrize("journal_error", [False, True])
def test_unreadable_files_are_reported(sample, monkeypatch, journal_error):
    workspace, journal, record = sample
    (journal / "op.json").write_text(json.dumps(record))
    def denied(*args, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(Path if journal_error else audit,
                        "read_text" if journal_error else "file_identity", denied)
    report = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread")
    assert (report["issues"][0]["status"] if journal_error else status(report)) == "unreadable"
    assert report["outcome"] == "issues"


@pytest.mark.parametrize("which", ["workspace", "journal"])
def test_missing_roots_remain_missing(sample, which):
    args = dict(zip(("workspace", "journal"), sample[:2]))
    missing = args[which] / "absent"
    args[which] = missing
    with pytest.raises(FileNotFoundError):
        audit.audit_journal(**args, thread_id="thread")
    assert not missing.exists()


@pytest.mark.parametrize("failure", ["read", "path_check", "disappeared", "directory"])
def test_entry_errors_preserve_later_records(sample, monkeypatch, failure):
    workspace, journal, record = sample
    for name in ("a.json", "b.json"):
        (journal / name).write_text(json.dumps(record))
    if failure == "directory":
        (journal / "a.json").unlink()
        (journal / "a.json").mkdir()
    else:
        obj, method = (audit, "checked_path") if failure == "path_check" else (Path, "read_text")
        original = getattr(obj, method)
        def broken(*args, **kwargs):
            if Path(args[-1]).name == "a.json":
                raise FileNotFoundError("gone") if failure == "disappeared" else PermissionError("denied")
            return original(*args, **kwargs)
        monkeypatch.setattr(obj, method, broken)
    report = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread")
    assert not report["selection_complete"] and report["counts"]["checked"] == 1
    assert report["issues"][0]["status"] == "unreadable"


def test_root_and_selection_errors(sample, monkeypatch):
    workspace, journal, _ = sample
    with pytest.raises(ValueError, match="thread_id"):
        audit.audit_records([], workspace=workspace, thread_id="")
    (journal / "file").touch()
    with pytest.raises(ValueError, match="directory"):
        audit.audit_journal(workspace=workspace, journal=journal / "file", thread_id="thread")
    def denied(*args, **kwargs):
        raise PermissionError("cannot list root")
    monkeypatch.setattr(audit.os, "scandir", denied)
    with pytest.raises(PermissionError):
        inspect(sample)
