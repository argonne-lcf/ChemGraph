"""Job journal preparation, waits, and publication without the service layer."""

import json

import pytest

from chemgraph.execution.jobs import LocalJobBackend, atomic_json
from chemgraph.runtime.artifacts import file_identity, output_manifest
from chemgraph.runtime.context import JobPending
from chemgraph.runtime.jobs import JobJournal, job_context, promote_outputs
from chemgraph.runtime.journal import FileOperationStore, decode_result
from chemgraph.runtime.policy import PreparedAction
from tests.runtime_helpers import approve, make_context, run_ase_call


@pytest.mark.asyncio
async def test_tampered_staged_input_is_blocked_before_dispatch(tmp_path):
    backend = LocalJobBackend(tmp_path / "backend")
    journal = JobJournal(FileOperationStore(tmp_path / "operations"), backend)
    context = make_context(tmp_path, journal=journal)
    call = run_ase_call()
    record = await journal.propose(context, call)
    action = PreparedAction.from_dict(record["prepared"])
    assert action.version == 2
    path = context.workspace / action.execution["root"] / action.inputs[0]["path"]
    path.write_text("tampered")
    await approve(context, record)
    result = await journal.execute(context, call, None)
    assert result.status == "error"
    assert not await backend.records()


@pytest.mark.asyncio
async def test_job_wait_is_not_inside_invoking_and_premature_wake_reuses_job(tmp_path):
    backend = LocalJobBackend(tmp_path / "backend")
    journal = JobJournal(FileOperationStore(tmp_path / "operations"), backend)
    context = make_context(tmp_path, journal=journal)
    call = run_ase_call()
    record = await journal.propose(context, call)
    await approve(context, record)
    for _ in range(2):
        with pytest.raises(JobPending):
            await journal.execute(context, call, None)
        assert (await journal.get(record["id"]))["state"] == "submitted"
    assert len(await backend.records()) == 1
    assert await journal.recover() == []


def test_partial_promotion_retry_preserves_overlapping_destinations(tmp_path, monkeypatch):
    from chemgraph.runtime import jobs
    context = make_context(tmp_path)
    root = context.workspace
    staged = []
    for name in ("h2_vib.0.traj", "h2_vib.1.traj"):
        (root / name).write_text("old")
        source = tmp_path / name
        source.write_text("new " + name)
        staged.append((source, root / name, {"source_path": name, **file_identity(source)}))
    (root / "h2_vib.9.traj").write_text("stale")
    replace = jobs.os.replace
    calls = []

    def fail_second(source, dest):
        calls.append(str(dest))
        if len(calls) == 2:
            raise OSError("Crash after first promotion")
        return replace(source, dest)

    monkeypatch.setattr(jobs.os, "replace", fail_second)
    cleanup = ["h2_vib.9.traj"]  # persisted stale targets minus promotion destinations
    with pytest.raises(OSError):
        promote_outputs(context, staged, cleanup)
    promote_outputs(context, staged, cleanup)
    assert (root / "h2_vib.0.traj").read_text() == "new h2_vib.0.traj"
    assert (root / "h2_vib.1.traj").read_text() == "new h2_vib.1.traj"
    assert not (root / "h2_vib.9.traj").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_public_error_paths_match_published_artifacts(tmp_path, cancelled):
    backend = LocalJobBackend(tmp_path / "backend")
    journal = JobJournal(FileOperationStore(tmp_path / "operations"), backend)
    context = make_context(tmp_path, journal=journal)
    call = run_ase_call()
    record = await journal.propose(context, call)
    await approve(context, record)
    with pytest.raises(JobPending):
        await journal.execute(context, call, None)
    action = PreparedAction.from_dict(record["prepared"])
    root = job_context(context, action).workspace
    (root / "energy.json").write_text('{"status":"partial"}')
    original = f"Thermochemistry failed; saved in {root / 'energy.json'}; input {root / action.inputs[0]['path']}"
    atomic_json(backend._directory(action.execution["job_id"]) / "result.json", {
        "job_id": action.execution["job_id"], "fingerprint": action.digest,
        "result": {"status": "failure", "message": original, "results_file": str(root / "energy.json")},
        "outputs": output_manifest(root, ["energy.json"]), "cleanup": [],
    })
    await journal.advance(context, cancelled=cancelled)
    final = await journal.get(record["id"])
    result = json.loads(decode_result(final["result"]["message"]).content)
    assert str(root) not in result["error"] and ".chemgraph-jobs" not in result["error"]
    public = final["result"]["artifacts"][0]["path"] if cancelled else "energy.json"
    assert f"saved in {public}; input h2.xyz" in result["error"]
    assert result["results_file"] == public
    assert final["returned_job"]["result"]["message"] == original
