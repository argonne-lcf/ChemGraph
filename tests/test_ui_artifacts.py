"""Tests for per-exchange artifact attribution (ui.artifacts)."""

import json
import os

from ui import artifacts


def _touch(path, mtime=None, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_snapshot_skips_manifest_and_handles_missing_dir(tmp_path):
    _touch(tmp_path / "a.xyz", mtime=10)
    _touch(tmp_path / "sub" / "b.csv", mtime=20)
    _touch(tmp_path / artifacts.MANIFEST_FILENAME)

    snapshot = artifacts.snapshot_mtimes(str(tmp_path))

    assert set(snapshot) == {"a.xyz", "sub/b.csv"}
    assert artifacts.snapshot_mtimes(None) == {}
    assert artifacts.snapshot_mtimes(str(tmp_path / "missing")) == {}


def test_collect_new_files_reports_only_this_runs_files(tmp_path):
    stale = _touch(tmp_path / "stale.xyz", mtime=10)
    modified = _touch(tmp_path / "rewritten.csv", mtime=20)
    before = artifacts.snapshot_mtimes(str(tmp_path))

    # Simulate a run: one new file, one rewritten file, one untouched file.
    _touch(tmp_path / "new.xyz", mtime=40)
    _touch(modified, mtime=50)

    changed = artifacts.collect_new_files(str(tmp_path), before)

    assert changed == ["new.xyz", "rewritten.csv"]
    assert stale.name not in changed


def test_collect_new_files_orders_by_mtime(tmp_path):
    before = artifacts.snapshot_mtimes(str(tmp_path))
    _touch(tmp_path / "later.png", mtime=200)
    _touch(tmp_path / "earlier.xyz", mtime=100)

    assert artifacts.collect_new_files(str(tmp_path), before) == [
        "earlier.xyz",
        "later.png",
    ]


def test_classify_artifacts_buckets_by_kind():
    kinds = artifacts.classify_artifacts(
        [
            "water_opt.xyz",
            "ir_spectrum_methanol.png",
            "frequencies_methanol.csv",
            "methanol_vib.3.traj",
            "report.html",
            "convergence.png",
            "output.json",
            "notes.txt",
        ]
    )

    assert kinds[artifacts.STRUCTURES] == ["water_opt.xyz"]
    assert kinds[artifacts.IR_PLOTS] == ["ir_spectrum_methanol.png"]
    assert kinds[artifacts.FREQUENCY_TABLES] == ["frequencies_methanol.csv"]
    assert kinds[artifacts.MODE_TRAJECTORIES] == ["methanol_vib.3.traj"]
    assert kinds[artifacts.REPORTS] == ["report.html"]
    assert kinds[artifacts.IMAGES] == ["convergence.png"]
    assert kinds[artifacts.DATA] == ["output.json"]
    assert kinds[artifacts.OTHER] == ["notes.txt"]


def test_manifest_round_trip(tmp_path):
    artifacts.append_manifest_entry(str(tmp_path), "optimize water", ["water.xyz"])
    artifacts.append_manifest_entry(
        str(tmp_path), "IR of methanol", ["ir_spectrum_methanol.png"]
    )

    entries = artifacts.load_manifest(str(tmp_path))

    assert entries == [
        {"query": "optimize water", "files": ["water.xyz"]},
        {"query": "IR of methanol", "files": ["ir_spectrum_methanol.png"]},
    ]


def test_load_manifest_tolerates_garbage(tmp_path):
    assert artifacts.load_manifest(None) == []
    assert artifacts.load_manifest(str(tmp_path)) == []

    manifest = tmp_path / artifacts.MANIFEST_FILENAME
    manifest.write_text("not json")
    assert artifacts.load_manifest(str(tmp_path)) == []

    manifest.write_text(json.dumps({"exchanges": [{"query": "q"}, "bad", 3]}))
    assert artifacts.load_manifest(str(tmp_path)) == []


def test_attach_artifacts_matches_by_order_and_query(tmp_path):
    artifacts.append_manifest_entry(str(tmp_path), "q1", ["a.xyz"])
    artifacts.append_manifest_entry(str(tmp_path), "q2", ["b.xyz"])
    history = [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]

    artifacts.attach_artifacts_to_history(history, str(tmp_path))

    assert history[0]["artifacts"] == ["a.xyz"]
    assert history[1]["artifacts"] == ["b.xyz"]
    assert "artifacts" not in history[2]


def test_attach_artifacts_stops_on_query_mismatch(tmp_path):
    artifacts.append_manifest_entry(str(tmp_path), "other query", ["a.xyz"])
    artifacts.append_manifest_entry(str(tmp_path), "q2", ["b.xyz"])
    history = [{"query": "q1"}, {"query": "q2"}]

    artifacts.attach_artifacts_to_history(history, str(tmp_path))

    # Positional correspondence is broken, so nothing is attached.
    assert "artifacts" not in history[0]
    assert "artifacts" not in history[1]
