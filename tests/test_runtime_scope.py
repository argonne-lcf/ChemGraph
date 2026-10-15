"""Worker-safe scope binding preserves confinement without runtime objects."""

import asyncio
from dataclasses import replace
import pickle

import pytest

from chemgraph.runtime import (
    WorkspaceScope, current_workspace, register_artifact, scoped_path, workspace_scope,
)
from chemgraph.runtime.context import standalone_log_record
from tests.runtime_helpers import make_context


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


@pytest.mark.asyncio
async def test_thread_scopes_are_independent(tmp_path):
    async def call(owner):
        scope = WorkspaceScope(owner, tmp_path / owner)
        with workspace_scope(scope):
            return await asyncio.to_thread(scoped_path, "same.xyz")
    assert await asyncio.gather(call("alice"), call("bob")) == [
        str(tmp_path / owner / "same.xyz") for owner in ("alice", "bob")]
    context = make_context(tmp_path)
    moved = replace(context, workspace=tmp_path / "moved")
    assert moved.scope.workspace == tmp_path / "moved"
