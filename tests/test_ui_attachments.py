"""Tests for chat file attachments (upload persistence and prompt notes)."""

import io

from ui import artifacts
from ui._pages import main_interface as main_ui
from ui.file_utils import persist_uploads


class _FakeUpload(io.BytesIO):
    def __init__(self, name, data: bytes):
        super().__init__(data)
        self.name = name


def test_persist_uploads_writes_files_and_returns_paths(tmp_path):
    uploads = [
        _FakeUpload("water.xyz", b"3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n"),
        _FakeUpload("data.csv", b"a,b\n1,2\n"),
    ]

    paths = persist_uploads(uploads, str(tmp_path))

    assert [p.split("/")[-1] for p in paths] == ["water.xyz", "data.csv"]
    assert (tmp_path / "water.xyz").read_bytes().startswith(b"3\nwater")
    assert (tmp_path / "data.csv").read_text() == "a,b\n1,2\n"


def test_persist_uploads_sanitizes_traversal_names(tmp_path):
    uploads = [_FakeUpload("../../etc/evil.xyz", b"x")]

    paths = persist_uploads(uploads, str(tmp_path))

    assert paths == [str(tmp_path / "evil.xyz")]
    assert (tmp_path / "evil.xyz").exists()
    assert not (tmp_path.parent / "etc").exists()


def test_persist_uploads_never_overwrites_existing_files(tmp_path):
    (tmp_path / "water.xyz").write_text("run output - do not clobber")
    uploads = [
        _FakeUpload("water.xyz", b"upload one"),
        _FakeUpload("water.xyz", b"upload two"),
    ]

    paths = persist_uploads(uploads, str(tmp_path))

    assert (tmp_path / "water.xyz").read_text() == "run output - do not clobber"
    assert [p.split("/")[-1] for p in paths] == ["water_1.xyz", "water_2.xyz"]
    assert (tmp_path / "water_1.xyz").read_bytes() == b"upload one"
    assert (tmp_path / "water_2.xyz").read_bytes() == b"upload two"


def test_persist_uploads_handles_empty_inputs(tmp_path):
    assert persist_uploads([], str(tmp_path)) == []
    assert persist_uploads(None, str(tmp_path)) == []
    assert persist_uploads([_FakeUpload("a.xyz", b"x")], None) == []


def test_attachment_note_lists_exact_paths():
    assert main_ui._attachment_note([]) == ""

    note = main_ui._attachment_note(["/run/turn_001/water.xyz", "/run/d.csv"])

    assert "attached the following file(s)" in note
    assert "- /run/turn_001/water.xyz" in note
    assert "- /run/d.csv" in note
    assert "exact paths" in note


def test_manifest_round_trips_attachment_names(tmp_path):
    artifacts.append_manifest_entry(
        str(tmp_path),
        "optimize the attached structure",
        ["turn_001/water_opt.traj"],
        attachments=["water.xyz"],
    )
    artifacts.append_manifest_entry(str(tmp_path), "plain query", ["a.json"])

    history = [
        {"query": "optimize the attached structure"},
        {"query": "plain query"},
    ]
    artifacts.attach_artifacts_to_history(history, str(tmp_path))

    assert history[0]["attachments"] == ["water.xyz"]
    assert history[0]["artifacts"] == ["turn_001/water_opt.traj"]
    assert "attachments" not in history[1]
