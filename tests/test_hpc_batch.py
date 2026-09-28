"""Scheduler/transfer lifecycle and native-tool regressions."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from unittest.mock import Mock

import globus_sdk
import httpx
import pytest

from chemgraph.execution import globus_transfer as transfer_module
from chemgraph.tools.alcf_iri_core import IRIClient, IRIRequestError
from chemgraph.tools.hpc.models import BatchRequest, HPCConfig, HPCTarget
from chemgraph.tools.hpc.service import HPCService
from chemgraph.tools.hpc.store import read_json, write_json
from chemgraph.tools.hpc.tools import create_hpc_registry


class Transfer:
    def __init__(self):
        self.state = "SUCCEEDED"
        self.calls = []

    def prepare_mapping(self, mapping, **kwargs):
        return dict(mapping), kwargs

    def submit_prepared(self, prepared):
        mapping, kwargs = prepared
        self.calls.append((mapping, kwargs))
        for source, destination in mapping.items():
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return f"transfer-{len(self.calls)}"

    def check_transfer_status(self, task_id):
        return {"task_id": task_id, "status": self.state}


class Scheduler:
    def __init__(self):
        self.submissions = []
        self.records = []
        self.failure = None
        self.cancelled = []
        self.finished = False

    def resolve_resource(self, name):
        return name + "-uuid"

    def prepare_submission(self, resource, spec):
        return resource, spec

    def submit_prepared(self, request):
        return self.submit(*request)

    def submit(self, resource, spec):
        self.submissions.append((resource, spec))
        self.records.append(
            {"id": "123.polaris", "job_spec": spec, "status": {"state": "queued"}}
        )
        if self.failure:
            raise self.failure
        return self.records[-1]

    def jobs(self, resource, **kwargs):
        if kwargs.get("offset", 0) or (self.finished and not kwargs.get("historical")):
            return []
        if kwargs.get("include_spec"):
            return self.records
        return [{k: v for k, v in record.items() if k != "job_spec"} for record in self.records]

    def status(self, resource, job_id, historical=False):
        if self.finished and not historical:
            raise IRIRequestError(404)
        return {
            "id": job_id,
            "status": {"state": "completed" if self.finished else "queued"},
        }

    def cancel(self, resource, job_id):
        self.cancelled.append((resource, job_id))


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_LOG_DIR", raising=False)
    local = tmp_path / "local"
    remote = tmp_path / "compute"
    root = local / "run"
    root.mkdir(parents=True)
    (root / "launch.sh").write_text("#!/bin/bash\ntrue\n")
    (root / "inputs").mkdir()
    (root / "inputs/water.xyz").write_text("3\nwater\nO 0 0 0\nH 0 0 1\nH 1 0 0\n")
    config = HPCConfig(
        targets={
            "polaris": HPCTarget(
                compute_resource="polaris",
                storage_resource="eagle",
                local_collection="local",
                remote_collection="eagle",
                local_root=str(local),
                local_collection_root=str(local),
                remote_root=str(remote),
                remote_collection_root=str(remote),
                project="project",
                queue="debug",
            )
        }
    )
    transfer, iri = Transfer(), Scheduler()
    service = HPCService(config, iri=iri, transfer_factory=lambda target: transfer)
    request = BatchRequest(target="polaris", launch_script="launch.sh")
    return service, root, transfer, iri, request


def stage(batch):
    service, root, *_ = batch
    return service.stage(str(root), "polaris", ["launch.sh", "inputs/water.xyz"])


@pytest.fixture
def sdk_transfer(batch, monkeypatch, tmp_path):
    service, *_ = batch
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = transfer_module._token_file(transfer_module._DEFAULT_CLIENT_ID)
    tokens = {"access_token": "private-access", "refresh_token": "private-refresh",
              "expires_at_seconds": 9999999999}
    transfer_module.GlobusTransferManager._save_tokens(cache, tokens)
    auth = Mock()
    monkeypatch.setattr(globus_sdk, "NativeAppAuthClient", Mock(return_value=auth))
    monkeypatch.setattr(globus_sdk.TransferClient, "get_submission_id", Mock(
        return_value={"value": "11111111-1111-1111-1111-111111111111"},
    ))
    submit = Mock(return_value={"task_id": "sdk-transfer"})
    monkeypatch.setattr(globus_sdk.TransferClient, "submit_transfer", submit)
    service.transfer_factory = service._transfer_manager
    return cache, tokens, auth, submit


@pytest.mark.parametrize("direction", ["stage", "retrieve"])
@pytest.mark.parametrize("failure", ["missing", "refresh", "payload", "payload_oserror"])
def test_transfer_preparation_can_retry_same_directory(
    batch, sdk_transfer, monkeypatch, direction, failure,
):
    service, root, *_ = batch
    cache, tokens, auth, submit = sdk_transfer
    if direction == "retrieve":
        stage(batch)
        submit.reset_mock()
        # An already cached client's refresh can fail in a long-running service.
        for manager in service._managers.values():
            if failure == "missing":
                manager._transfer_client = None
            elif failure == "refresh":
                manager._transfer_client.authorizer.expires_at = 1
    if failure == "missing":
        cache.unlink()
    elif failure == "refresh":
        transfer_module.GlobusTransferManager._save_tokens(cache, {**tokens, "expires_at_seconds": 1})
        auth.oauth2_refresh_token.side_effect = RuntimeError("private-refresh")
    payload_class = globus_sdk.TransferData
    if failure.startswith("payload"):
        error = OSError if failure == "payload_oserror" else TypeError
        monkeypatch.setattr(globus_sdk, "TransferData", Mock(side_effect=error("private-access")))
    tool = create_hpc_registry({}, names=["hpc_transfer_files"], service=service).get("hpc_transfer_files")
    args = {"run_dir": str(root), "direction": direction, "target": "polaris",
            "files": ["launch.sh"] if direction == "stage" else ["outputs/result.json"]}
    result = tool.invoke(args)
    assert result["phase"] == "prepare" and result["retry_safe"] is True
    assert "private-" not in str(result)
    if failure in ("missing", "refresh"):
        assert result["error"] == "authentication_required"
        assert "python -m chemgraph.execution.globus_transfer" in result["message"]
    else:
        assert result["type"] == error.__name__
    submit.assert_not_called()
    assert not (root / "retrieved").exists()
    assert not (root / ".hpc-retrievals.json").exists()
    if direction == "stage":
        assert not (root / "run.json").exists()
        assert not (root / ".hpc-inputs").exists()
    if failure == "refresh":
        assert all(manager._transfer_client is None for manager in service._managers.values())
        assert auth.oauth2_refresh_token.call_count == 1
    # Simulate terminal login replacing the token cache, without replacing the service.
    transfer_module.GlobusTransferManager._save_tokens(cache, tokens)
    monkeypatch.setattr(globus_sdk, "TransferData", payload_class)

    def submitted(payload):
        assert isinstance(payload, payload_class)
        if direction == "stage":
            assert read_json(root / "run.json")["staging"] == "unknown"
            assert (root / ".hpc-inputs/launch.sh").read_bytes() == (root / "launch.sh").read_bytes()
        else:
            assert read_json(root / ".hpc-retrievals.json")[-1]["transfer_id"] is None
            assert (root / "retrieved/outputs").is_dir()
            assert payload["source_endpoint"] == "eagle"
            assert payload["destination_endpoint"] == "local"
        return {"task_id": "sdk-transfer"}

    submit.side_effect = submitted
    assert tool.invoke(args)["transfer_id"] == "sdk-transfer"
    submit.assert_called_once()
    if failure == "refresh":
        assert auth.oauth2_refresh_token.call_count == 1
    evidence = (root / "run.json").read_text()
    if direction == "retrieve":
        evidence += (root / ".hpc-retrievals.json").read_text()
    assert "private-" not in evidence


@pytest.mark.parametrize("direction", ["stage", "retrieve"])
@pytest.mark.parametrize("failure", ["timeout", "interrupt", "save_id"])
def test_uncertain_transfer_preserves_evidence(batch, sdk_transfer, monkeypatch, direction, failure):
    import chemgraph.tools.hpc.service as module

    service, root, *_ = batch
    *_, submit = sdk_transfer
    if direction == "retrieve":
        stage(batch)
        submit.reset_mock()
    evidence = root / ("run.json" if direction == "stage" else ".hpc-retrievals.json")
    if failure == "save_id":
        original = module.write_json

        def fail_save(path, value):
            record = value if direction == "stage" else value[-1]
            if path == evidence and record.get("transfer_id"):
                raise OSError("private-access")
            original(path, value)

        monkeypatch.setattr(module, "write_json", fail_save)
    else:
        submit.side_effect = SystemExit("interrupted") if failure == "interrupt" else TimeoutError("private-access")
    tool = create_hpc_registry({}, names=["hpc_transfer_files"], service=service).get("hpc_transfer_files")
    args = {"run_dir": str(root), "direction": direction, "target": "polaris", "files": ["launch.sh"]}
    if failure == "interrupt":
        with pytest.raises(SystemExit):
            tool.invoke(args)
    else:
        result = tool.invoke(args)
        assert result["phase"] == "submit" and result["retry_safe"] is False
        assert result["error"] == "operation_failed"
        assert "private-" not in str(result)
    saved = evidence.read_bytes()
    record = read_json(evidence)
    assert (record if direction == "stage" else record[-1])["transfer_id"] is None
    assert tool.invoke(args)["error"] == "invalid_run"
    submit.assert_called_once()
    assert evidence.read_bytes() == saved
    if direction == "stage":
        assert service.transfer_status(str(root))["state"] == "transfer_unknown"
        assert evidence.read_bytes() == saved


def test_existing_unknown_transfer_status_is_actionable_and_read_only(batch):
    service, root, transfer, *_ = batch
    manifest = stage(batch)
    manifest.update(transfer_id=None, staging="unknown")
    write_json(root / "run.json", manifest)
    saved = (root / "run.json").read_bytes()
    tool = create_hpc_registry({}, names=["hpc_transfer_status"], service=service).get("hpc_transfer_status")
    result = tool.invoke({"run_dir": str(root)})
    assert result["state"] == "transfer_unknown" and result["transfer_id"] is None
    assert result["retry_safe"] is False and "do not restage" in result["message"]
    for transfer_id in ("unrecorded", ""):
        assert tool.invoke({"run_dir": str(root), "transfer_id": transfer_id})["error"] == "invalid_run"
    with pytest.raises(ValueError, match="already attempted"):
        stage(batch)
    assert (root / "run.json").read_bytes() == saved
    assert len(transfer.calls) == 1


def test_stage_preserves_structure_and_freezes_identity(batch):
    service, root, transfer, iri, request = batch
    manifest = stage(batch)
    remote = Path(manifest["remote_directory"])
    assert (remote / "inputs/water.xyz").read_bytes() == (
        root / "inputs/water.xyz"
    ).read_bytes()
    assert not (remote / "water.xyz").exists()
    assert manifest["target_snapshot"]["compute_resource"] == "polaris-uuid"
    assert service.submit(str(root), request)["state"] == "accepted"
    assert service.submit(str(root), request)["job_id"] == "123.polaris"
    assert len(iri.submissions) == 1
    assert iri.submissions[0][1]["environment"] == {}
    assert len(transfer.calls) == 1
    with pytest.raises(ValueError, match="fresh"):
        stage(batch)
    with pytest.raises(ValueError, match="changed"):
        service.submit(
            str(root), request.model_copy(update={"arguments": ["different"]})
        )


@pytest.mark.parametrize("state", ["accepted", "rejected", "submission_unknown", "prepared"])
def test_legacy_submission_replay_preserves_evidence(batch, state):
    service, root, transfer, iri, request = batch
    manifest = stage(batch)
    # Build evidence through the original JobSpec, then restart with current code.
    original = service._jobspec
    service._jobspec = lambda *args: {
        **original(*args),
        "environment": {"CHEMGRAPH_LOG_DIR": manifest["remote_directory"]},
    }
    service.submit(str(root), request)
    path = root / "submission.json"
    evidence = read_json(path)
    evidence["state"] = state
    if state != "accepted":
        evidence.pop("job_id")
    write_json(path, evidence)
    before = path.read_bytes()
    restarted = HPCService({}, iri=iri, transfer_factory=lambda target: transfer)
    result = restarted.submit(str(root), request)
    assert result["state"] == ("submission_unknown" if state == "prepared" else state)
    assert path.read_bytes() == before
    assert len(iri.submissions) == 1
    status = restarted.status(str(root))
    if state == "rejected":
        assert status["state"] == "rejected"
    else:
        assert status["job_id"] == "123.polaris"


def test_legacy_pre_marker_recovery_uses_original_spec(batch, monkeypatch):
    service, root, transfer, iri, request = batch
    manifest = stage(batch)
    import chemgraph.tools.hpc.service as module

    original_spec = service._jobspec
    service._jobspec = lambda *args: {
        **original_spec(*args),
        "environment": {"CHEMGRAPH_LOG_DIR": manifest["remote_directory"]},
    }
    with monkeypatch.context() as patch:
        patch.setattr(module, "mark_started", Mock(side_effect=SystemExit))
        with pytest.raises(SystemExit):
            service.submit(str(root), request)
    before = read_json(root / "submission.json")
    restarted = HPCService({}, iri=iri, transfer_factory=lambda target: transfer)
    assert restarted.submit(str(root), request)["state"] == "accepted"
    after = read_json(root / "submission.json")
    assert after["intent"] == before["intent"]
    assert after["created_at"] == before["created_at"]
    assert iri.submissions == [("polaris-uuid", before["intent"]["jobspec"])]


@pytest.mark.parametrize("change", ["environment", "extra_variable", "arguments", "request"])
def test_legacy_compatibility_rejects_other_spec_changes(batch, change):
    service, root, _, iri, request = batch
    manifest = stage(batch)
    service.submit(str(root), request)
    path = root / "submission.json"
    evidence = read_json(path)
    spec = evidence["intent"]["jobspec"]
    spec["environment"] = {"CHEMGRAPH_LOG_DIR": manifest["remote_directory"]}
    if change == "environment":
        spec["environment"]["CHEMGRAPH_LOG_DIR"] = "/different"
    elif change == "extra_variable":
        spec["environment"]["OTHER"] = "value"
    elif change == "arguments":
        spec["arguments"].append("different")
    else:
        request = request.model_copy(update={"queue": "different"})
    write_json(path, evidence)
    with pytest.raises(ValueError, match="specification changed"):
        service.submit(str(root), request)
    assert len(iri.submissions) == 1


@pytest.mark.parametrize("state", ["ACTIVE", "FAILED", "INACTIVE"])
def test_incomplete_transfer_never_submits(batch, state):
    service, root, transfer, iri, request = batch
    stage(batch)
    transfer.state = state
    with pytest.raises(ValueError, match="complete"):
        service.submit(str(root), request)
    assert not iri.submissions
    assert not (root / "submission.started").exists()


@pytest.mark.parametrize("after_submit", [False, True])
def test_changed_inputs_require_fresh_run(batch, after_submit):
    service, root, _, iri, request = batch
    stage(batch)
    if after_submit:
        service.submit(str(root), request)
    (root / "inputs/water.xyz").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        service.submit(str(root), request)
    assert len(iri.submissions) == int(after_submit)


def test_concurrent_submission_and_new_session_monitoring(batch):
    service, root, transfer, iri, request = batch
    stage(batch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: service.submit(str(root), request), range(2)))
    assert len(iri.submissions) == 1
    assert all(item["job_id"] == "123.polaris" for item in results)
    # No current configuration: saved target and collection identity are sufficient.
    restarted = HPCService({}, iri=iri, transfer_factory=lambda target: transfer)
    iri.finished = True
    assert restarted.status(str(root))["scheduler"]["state"] == "completed"
    assert restarted.status(str(root))["scientific_success"] is None
    restarted.cancel(str(root), "123.polaris")
    assert iri.cancelled == [("polaris-uuid", "123.polaris")]
    with pytest.raises(ValueError):
        restarted.cancel(str(root), "123")


@pytest.mark.parametrize(
    "failure", [RuntimeError("token must not be saved"), SystemExit("crash")]
)
@pytest.mark.parametrize("finished", [False, True])
def test_lost_response_and_crash_reconcile_without_resubmit(batch, failure, finished):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = failure
    if isinstance(failure, SystemExit):
        with pytest.raises(SystemExit):
            service.submit(str(root), request)
    else:
        assert service.submit(str(root), request)["state"] == "submission_unknown"
    assert service.submit(str(root), request)["state"] == "submission_unknown"
    assert "token must not be saved" not in (root / "submission.json").read_text()
    iri.finished = finished
    assert service.status(str(root))["job_id"] == "123.polaris"
    assert len(iri.submissions) == 1


@pytest.mark.parametrize("records", ["missing", "ambiguous", "wrong_directory", "missing_spec"])
def test_missing_or_ambiguous_history_stays_unknown(batch, records):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = RuntimeError("lost")
    service.submit(str(root), request)
    if records == "missing":
        iri.records = []
    elif records == "ambiguous":
        iri.records.append({**iri.records[0], "id": "456.polaris"})
    elif records == "missing_spec":
        iri.records[0].pop("job_spec")
    else:
        iri.records[0]["job_spec"]["directory"] = "/another/run"
    assert service.status(str(root))["state"] == "submission_unknown"
    assert len(iri.submissions) == 1


@pytest.mark.parametrize("historical", [False, True])
def test_reconciliation_requests_specs_from_iri(batch, historical):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = RuntimeError("lost response")
    service.submit(str(root), request)
    calls = []

    def respond(http_request):
        calls.append(http_request)
        if http_request.method == "POST":
            records = iri.records
            if historical and http_request.url.params["historical"] != "true":
                records = []
            if http_request.url.params.get("include_spec") != "true":
                records = [{k: v for k, v in row.items() if k != "job_spec"} for row in records]
            return httpx.Response(200, json=records)
        return httpx.Response(200, json=iri.records[0])

    service.iri = IRIClient(transport=httpx.MockTransport(respond), headers=lambda: {})
    assert service.status(str(root))["job_id"] == "123.polaris"
    searches = [call for call in calls if call.method == "POST"]
    assert len(searches) == 2
    assert all(call.url.params["include_spec"] == "true" for call in searches)
    assert all("/compute/status/" in call.url.path for call in calls)
    assert len(iri.submissions) == 1


def test_rejection_and_pre_marker_crash(batch, monkeypatch):
    service, root, _, iri, request = batch
    stage(batch)
    import chemgraph.tools.hpc.service as module

    original = module.mark_started
    monkeypatch.setattr(module, "mark_started", Mock(side_effect=SystemExit))
    with pytest.raises(SystemExit):
        service.submit(str(root), request)
    assert not iri.submissions
    monkeypatch.setattr(module, "mark_started", original)
    iri.failure = IRIRequestError(422)
    assert service.submit(str(root), request)["state"] == "rejected"
    assert service.submit(str(root), request)["state"] == "rejected"
    assert len(iri.submissions) == 1


def test_authentication_failure_can_retry_same_run_without_resubmission(batch):
    service, root, _, _, request = batch
    stage(batch)
    calls = []

    def respond(http_request):
        assert (root / "submission.started").exists()
        assert read_json(root / "submission.json")["state"] == "prepared"
        assert http_request.headers["Authorization"] == "Bearer test-credential"
        calls.append(http_request)
        return httpx.Response(200, json={"id": "123.polaris"})

    headers = Mock(side_effect=RuntimeError("secret provider error"))
    service.iri = IRIClient(transport=httpx.MockTransport(respond), headers=headers)
    tool = create_hpc_registry({}, names=["hpc_submit_job"], service=service).get("hpc_submit_job")
    args = {"run_dir": str(root), "request": request.model_dump()}
    result = tool.invoke(args)
    assert result["error"] == "authentication_required"
    assert "same run directory" in result["message"]
    assert "secret" not in str(result)
    assert not calls
    assert not (root / "submission.started").exists()
    assert not (root / "submission.json").exists()
    assert service.status(str(root))["state"] == "not_submitted"

    headers.side_effect = [{"Authorization": "Bearer test-credential"}]
    assert tool.invoke(args)["state"] == "accepted"
    # The single prepared credential is sufficient; neither sending nor an
    # accepted repeat should call the provider again.
    assert tool.invoke(args)["job_id"] == "123.polaris"
    assert len(calls) == 1
    assert headers.call_count == 2
    assert "test-credential" not in (root / "submission.json").read_text()


def test_invalid_submission_payload_does_not_mark_attempt(batch, monkeypatch):
    service, root, _, _, request = batch
    stage(batch)
    service.iri = IRIClient(headers=lambda: {})
    original = service._jobspec
    monkeypatch.setattr(service, "_jobspec", lambda *args: {**original(*args), "bad": object()})
    with pytest.raises(TypeError):
        service.submit(str(root), request)
    assert not (root / "submission.started").exists()
    assert not (root / "submission.json").exists()


def test_prepared_submission_transport_failure_remains_unknown(batch):
    service, root, _, _, request = batch
    stage(batch)
    calls = []

    def fail(http_request):
        calls.append(http_request)
        raise httpx.ReadTimeout("secret transport error")

    service.iri = IRIClient(transport=httpx.MockTransport(fail), headers=lambda: {})
    assert service.submit(str(root), request)["state"] == "submission_unknown"
    assert service.submit(str(root), request)["state"] == "submission_unknown"
    assert len(calls) == 1
    assert (root / "submission.started").exists()
    assert "secret" not in (root / "submission.json").read_text()


def test_retrieval_mapping_and_overwrite(batch):
    service, root, _, _, _ = batch
    manifest = stage(batch)
    output = Path(manifest["remote_directory"]) / "result.json"
    output.write_text('{"success":true}')
    record = service.retrieve(str(root), ["result.json"])
    assert (root / "retrieved/result.json").read_text() == output.read_text()
    assert (
        service.transfer_status(str(root), record["transfer_id"])["status"]
        == "SUCCEEDED"
    )
    with pytest.raises(ValueError, match="exists"):
        service.retrieve(str(root), ["result.json"])
    service.retrieve(str(root), ["result.json"], overwrite=True)


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "run.json", ".hpc-inputs/script"]
)
def test_invalid_paths_do_not_start_transfer(batch, name):
    service, root, transfer, *_ = batch
    with pytest.raises(ValueError):
        service.stage(str(root), "polaris", [name])
    assert not transfer.calls


def test_symlinks_and_manifest_version(batch):
    service, root, transfer, *_ = batch
    (root / "link").symlink_to(root / "launch.sh")
    with pytest.raises(ValueError, match="Symlink"):
        service.stage(str(root), "polaris", ["link"])
    assert not transfer.calls
    stage(batch)
    manifest = read_json(root / "run.json")
    manifest["version"] = 999
    write_json(root / "run.json", manifest)
    with pytest.raises(ValueError, match="version"):
        service.status(str(root))


def test_two_catalogs_and_empty_restriction(batch):
    service, *_ = batch
    one = create_hpc_registry(service.config, names=["hpc_list_targets"])
    two = create_hpc_registry({"targets": {}}, names=["hpc_list_targets"])
    assert "polaris" in one.get("hpc_list_targets").invoke({})["targets"]
    assert two.get("hpc_list_targets").invoke({})["targets"] == {}
    assert create_hpc_registry(service.config, names=[]).names() == ()


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize(
    "action", ["hpc_transfer_files", "hpc_submit_job", "hpc_cancel_job"]
)
def test_native_reviews_have_no_rejected_side_effects(batch, decision, action):
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.types import Command
    from chemgraph.graphs.deep_agent import construct_deep_agent_graph
    from tests.test_registry_middleware import CatalogModel, call

    service, root, transfer, iri, request = batch
    if action != "hpc_transfer_files":
        stage(batch)
    if action == "hpc_cancel_job":
        service.submit(str(root), request)
    args = {
        "hpc_transfer_files": {
            "run_dir": str(root),
            "target": "polaris",
            "direction": "stage",
            "files": ["launch.sh"],
        },
        "hpc_submit_job": {"run_dir": str(root), "request": request.model_dump()},
        "hpc_cancel_job": {"run_dir": str(root), "job_id": "123.polaris"},
    }[action]
    registry = create_hpc_registry(service.config, names=[action], service=service)
    graph = construct_deep_agent_graph(
        CatalogModel(
            responses=[
                call("load_tools", names=[action]),
                call(action, **args),
                AIMessage(content="done"),
            ]
        ),
        tool_registry=registry,
        discover_skills=False,
    )
    config = {"configurable": {"thread_id": "test"}}
    before = (len(transfer.calls), len(iri.submissions), len(iri.cancelled))
    state = graph.invoke(
        {"messages": [HumanMessage(content="Run the HPC action")]}, config
    )
    assert state["__interrupt__"]
    assert before == (len(transfer.calls), len(iri.submissions), len(iri.cancelled))
    graph.invoke(Command(resume={"decisions": [{"type": decision}]}), config)
    after = (len(transfer.calls), len(iri.submissions), len(iri.cancelled))
    assert (after == before) == (decision == "reject")
    if action == "hpc_submit_job" and decision == "reject":
        assert not (root / "submission.started").exists()


def test_explicit_config_and_cli_restrictions(batch, tmp_path, monkeypatch):
    import importlib
    import toml

    cli = importlib.import_module("chemgraph.cli.main")
    service, root, *_ = batch
    config = tmp_path / "selected.toml"
    config.write_text(
        toml.dumps(
            {
                "general": {"workflow": "deep_agent"},
                "hpc": service.config.model_dump(exclude_none=True),
            }
        )
    )
    received = {}
    monkeypatch.setattr(
        cli, "interactive_mode", lambda **kwargs: received.update(kwargs)
    )
    args = cli.create_argument_parser().parse_args(
        [
            "run",
            "--interactive",
            "--config",
            str(config),
            "--tool",
            "hpc_list_targets",
        ]
    )
    cli._handle_run(args)
    registry = received["deepagent_tool_registry"]
    assert registry.names() == ("hpc_list_targets",)
    assert registry.get("hpc_list_targets").invoke({})["targets"]["polaris"][
        "local_root"
    ] == str(root.parent)


def test_collection_paths_are_not_compute_or_host_paths(tmp_path):
    target = HPCTarget(
        compute_resource="polaris",
        storage_resource="eagle",
        local_collection="local",
        remote_collection="eagle",
        local_root=str(tmp_path),
        local_collection_root="/source",
        remote_collection_root="/project",
        remote_root="/eagle/project",
        project="project",
        queue="debug",
    )
    assert target.local_path(tmp_path / "run/input.xyz") == "/source/run/input.xyz"
    assert (
        target.collection_path("/eagle/project/run/input.xyz")
        == "/project/run/input.xyz"
    )
    with pytest.raises(ValueError):
        target.collection_path("/other-project/file")


def test_concurrent_staging_has_one_transfer(batch):
    _, _, transfer, _, _ = batch

    def attempt(_):
        try:
            stage(batch)
            return "staged"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, range(2))) == ["rejected", "staged"]
    assert len(transfer.calls) == 1


def test_saved_submission_operation_is_polled_without_resubmission(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.submit = Mock(return_value={"task_id": "operation"})
    iri.task = Mock(
        side_effect=[
            {"id": "operation", "status": "active", "result": None},
            {"id": "operation", "status": "completed", "result": {"id": "123.polaris"}},
        ]
    )
    assert service.submit(str(root), request)["state"] == "submission_unknown"
    assert service.status(str(root))["state"] == "submission_unknown"
    assert service.status(str(root))["job_id"] == "123.polaris"
    iri.submit.assert_called_once()


def test_search_limit_and_unknown_shapes_cannot_resolve_acceptance(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = RuntimeError("lost")
    service.submit(str(root), request)
    iri.jobs = Mock(return_value=[iri.records[0]] * 100)
    assert service.status(str(root))["state"] == "submission_unknown"
    assert iri.jobs.call_count == 10
    iri.jobs = Mock(return_value={"unexpected": iri.records})
    assert service.status(str(root))["state"] == "submission_unknown"


def test_source_changed_during_snapshot_never_transfers(batch, monkeypatch):
    service, root, transfer, *_ = batch
    original = shutil.copyfile

    def changing_copy(source, destination):
        result = original(source, destination)
        Path(source).write_text("modified while copying")
        return result

    monkeypatch.setattr(shutil, "copyfile", changing_copy)
    with pytest.raises(ValueError, match="changed"):
        stage(batch)
    assert not transfer.calls


def test_custom_review_policy_is_preserved(batch):
    from langchain_core.messages import AIMessage, HumanMessage
    from chemgraph.graphs.deep_agent import construct_deep_agent_graph
    from tests.test_registry_middleware import CatalogModel, call

    service, root, _, iri, request = batch
    stage(batch)
    graph = construct_deep_agent_graph(
        CatalogModel(
            responses=[
                call("load_tools", names=["hpc_submit_job"]),
                call("hpc_submit_job", run_dir=str(root), request=request.model_dump()),
                AIMessage(content="done"),
            ]
        ),
        tool_registry=create_hpc_registry(
            {}, names=["hpc_submit_job"], service=service
        ),
        interrupt_on={"hpc_submit_job": False},
        discover_skills=False,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="Submit")]},
        {"configurable": {"thread_id": "trusted"}},
    )
    assert "__interrupt__" not in result and len(iri.submissions) == 1


def test_monitoring_cannot_retarget_a_prepared_run(batch):
    service, root, _, iri, request = batch
    stage(batch)
    service.submit(str(root), request)
    manifest = read_json(root / "run.json")
    manifest["target_snapshot"]["compute_resource"] = "different-machine"
    write_json(root / "run.json", manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        service.status(str(root))
    with pytest.raises(ValueError, match="manifest changed"):
        service.cancel(str(root), "123.polaris")
    assert not iri.cancelled
