"""Immutable uploads and storage accounting without starting an HTTP server."""

import pytest

from chemgraph.mcp.service.files import Files, QuotaExceeded
from chemgraph.runtime.sqlite_store import initialize_database
from tests.mcp_service_helpers import config


def test_replacing_workspace_input_preserves_published_versions_and_owner(tmp_path):
    settings = config(tmp_path)
    initialize_database(settings.database)
    files = Files(settings)
    first = files.ingest("alice", "input.xyz", [b"first version"], text=True)
    second = files.ingest("alice", "input.xyz", [b"second version"], text=True)
    assert first["artifact_id"] != second["artifact_id"]
    assert files.read("alice", first["artifact_id"])["content"] == "first version"
    assert files.read("alice", second["artifact_id"])["content"] == "second version"
    assert files.path("alice", "input.xyz").read_text() == "second version"
    assert files.list("bob") == []
    for identifier in (first["artifact_id"], "unknown"):
        with pytest.raises(FileNotFoundError, match="^Unknown artifact.$"):
            files.get("bob", identifier)


def test_storage_quota_includes_published_copies_and_failed_upload_is_atomic(tmp_path):
    settings = config(tmp_path, max_storage_bytes=256)
    initialize_database(settings.database)
    files = Files(settings)
    saved = files.ingest("alice", "input.xyz", [b"original"], text=True)
    assert files.usage("alice") == 2 * len(b"original")
    files.path("alice", "occupied.bin").write_bytes(b"x" * 240)
    with pytest.raises(QuotaExceeded, match="storage quota"):
        files.ingest("alice", "input.xyz", [b"replacement"], text=True)
    assert files.path("alice", "input.xyz").read_bytes() == b"original"
    assert files.read("alice", saved["artifact_id"])["content"] == "original"
    assert not list(files.workspace("alice").glob(".service-upload-*"))
