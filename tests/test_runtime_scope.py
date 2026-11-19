"""Worker-safe scope binding preserves confinement without runtime objects."""

import pickle

import pytest

from chemgraph.runtime import (
    WorkspaceScope, current_workspace, register_artifact, scoped_path, workspace_scope,
)
from chemgraph.runtime.context import standalone_log_record


def test_worker_scope_snapshot_binding_and_logging(tmp_path):
    scope = pickle.loads(pickle.dumps(WorkspaceScope("alice", tmp_path)))
    path = ".chemgraph-inputs/digest/water.xyz"
    outputs = set()
    with workspace_scope(scope):
        assert not standalone_log_record(None)
        with pytest.raises(PermissionError):
            scoped_path(path)
        with workspace_scope(scope, inputs=[path], outputs=outputs):
            assert scoped_path(path) == str(tmp_path / path)
            register_artifact("out.json")
        with pytest.raises(PermissionError):
            scoped_path(path)
    assert outputs == {"out.json"} and current_workspace() is None
    assert standalone_log_record(None)
