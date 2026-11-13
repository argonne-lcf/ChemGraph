"""Scoped artifact inspection is read-only and resilient to individual failures."""

import copy
import json
import os

import pytest

from chemgraph.runtime import FileOperationStore, OperationJournal
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
        pytest.fail("Audit must not initialize, execute, or decode runtime state")
    for obj, name in ((FileOperationStore, "__init__"), (OperationJournal, "execute"),
                      (journal_module, "decode_result")):
        monkeypatch.setattr(obj, name, forbidden)
    report = inspect(sample, records=iter([record]), inputs=True)
    disk = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread", include_inputs=True)
    assert report["outcome"] == disk["outcome"] == "verified"
    assert report["counts"]["checked"] == 2
    assert report["records"][0]["execution_outcome"] == "unknown"
    assert report["records"][0]["scientific_acceptance"] == "unassessed"
    assert record == original and inventory(workspace.parent) == before


def test_corruption_and_optional_input_scope(sample):
    workspace, _, record = sample
    path = workspace / record["prepared"]["inputs"][0]["path"]
    saved = path.stat()
    path.write_bytes(b"change")  # Same size and mtime still require rehashing.
    os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    assert inspect(sample)["outcome"] == "verified"
    report = inspect(sample, inputs=True)
    assert report["records"][0]["descriptors"][1]["status"] == "mismatch"
    assert report["counts"]["checked"] == 2 and report["selection_complete"]


@pytest.mark.parametrize("change,expected", [
    ({"path": ".chemgraph-artifacts/missing"}, "missing"),
    ({"size_bytes": 0}, "mismatch"),
    ({"sha256": "invalid"}, "malformed_descriptor"),
    ({"size_bytes": True}, "malformed_descriptor"),
    ({"path": ".chemgraph-artifacts/../outside"}, "invalid_path"),
    ({"path": "C:/outside"}, "invalid_path"),
])
def test_bad_descriptor_preserves_valid_sibling(sample, change, expected):
    descriptors = sample[2]["result"]["artifacts"]
    descriptors.insert(0, {**descriptors[0], **change})
    report = inspect(sample)
    assert [d["status"] for d in report["records"][0]["descriptors"]] == [expected, "match"]
    assert report["outcome"] == "issues"


@pytest.mark.parametrize("kind", ["output", "journal"])
def test_symlinks_rejected(sample, kind):
    workspace, journal, record = sample
    target = journal / "op.json" if kind == "journal" else workspace / record["result"]["artifacts"][0]["path"]
    moved = workspace.parent / "target"
    moved.write_text(json.dumps(record))
    target.unlink(missing_ok=True)
    try:
        target.symlink_to(moved)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    report = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread") if kind == "journal" else inspect(sample)
    finding = report["issues"][0] if kind == "journal" else report["records"][0]["descriptors"][0]
    assert finding["status"] == "invalid_path" and report["outcome"] == "issues"


@pytest.mark.parametrize("bad", [b"{", b"{}"])
def test_bad_journal_entry_does_not_hide_valid_records(sample, bad):
    workspace, journal, record = sample
    (journal / "a.json").write_bytes(bad)
    (journal / "b.json").write_text(json.dumps(record))
    report = audit.audit_journal(workspace=workspace, journal=journal, thread_id="thread")
    assert report["outcome"] == "issues" and not report["selection_complete"]
    assert report["counts"]["checked"] == 1 and report["issues"][0]["origin"] == "a.json"


def test_missing_container_is_not_an_empty_selection(sample):
    sample[2]["result"] = {}
    report = inspect(sample, inputs=True)
    assert report["outcome"] == "issues" and not report["selection_complete"]
    assert report["counts"]["checked"] == 1  # Valid input still checked.
    sample[2]["result"]["artifacts"] = []
    assert inspect(sample)["outcome"] == "no_descriptors"


def test_thread_selection_and_unfinalized_records(sample):
    record = sample[2]
    skipped = {**record, "state": "returned", "result": None}
    excluded = {"thread_id": "other", "result": "do not inspect"}
    report = inspect(sample, records=[excluded, skipped, record])
    assert report["outcome"] == "verified"
    assert report["counts"]["excluded"] == report["counts"]["skipped"] == 1
    assert inspect(sample, records=[excluded, skipped])["outcome"] == "no_descriptors"
    assert inspect(sample, records=[{**record, "state": "unknown"}])["outcome"] == "issues"


def test_iterator_failure_retains_completed_findings(sample):
    def broken():
        yield sample[2]
        raise OSError("custom store unavailable")
    report = inspect(sample, records=broken())
    assert report["outcome"] == "issues" and report["counts"]["checked"] == 1
    assert not report["selection_complete"]


def test_unreadable_file_is_reported(sample, monkeypatch):
    def denied(*args):
        raise PermissionError("denied")
    monkeypatch.setattr(audit, "file_identity", denied)
    assert status(inspect(sample)) == "unreadable"


def test_missing_root_stays_missing(sample):
    workspace, journal, _ = sample
    missing = journal / "absent"
    with pytest.raises(FileNotFoundError):
        audit.audit_journal(workspace=workspace, journal=missing, thread_id="thread")
    assert not missing.exists()
