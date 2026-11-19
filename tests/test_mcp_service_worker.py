"""Worker report failures preserve outputs and remain JSON serializable."""

import json
from types import SimpleNamespace

from chemgraph.mcp.service import worker
from chemgraph.runtime.context import WorkspaceScope, register_artifact


def test_nonfinite_worker_result_keeps_registered_outputs(tmp_path, monkeypatch):
    def invoke(arguments):
        (tmp_path / "out.json").write_text('{"energy": null}')
        register_artifact("out.json")
        return {"energy": float("nan")}

    monkeypatch.setattr(worker, "chemistry_tools", lambda: {
        "run_ase": SimpleNamespace(invoke=invoke),
    })
    action = SimpleNamespace(tool="run_ase", inputs=[], arguments={})
    report = worker.execute(action, WorkspaceScope("alice", tmp_path))
    assert report["error"] and report["result"]["error_type"] == "ValueError"
    assert report["registered"] == ["out.json"]
    json.dumps(report, allow_nan=False)
