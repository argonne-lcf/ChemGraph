"""Scheduler/transfer lifecycle and native-tool regressions."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import globus_sdk
import httpx
import pytest

from chemgraph.execution import globus_transfer as transfer_module
from chemgraph.tools.alcf_iri_core import IRIClient, IRIRequestError
from chemgraph.tools.hpc.models import BatchRequest, HPCConfig, HPCTarget, Resources
from chemgraph.tools.hpc.service import HPCService
from chemgraph.tools.hpc.store import read_json, write_json
from chemgraph.tools.hpc.tools import create_hpc_registry


class OfflineIRIClient(IRIClient):
    """Real request handling with the fake scheduler's resource names."""

    def resolve_resource(self, name):
        return name + "-uuid"


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
        self.inspections = []
        self.finished = False

    def resolve_resource(self, name):
        return name + "-uuid"

    def prepare_submission(self, resource, spec):
        return resource, spec

    def submit_prepared(self, request):
        return self.submit(*request)

    def submit(self, resource, spec):
        self.submissions.append((resource, spec))
        if isinstance(self.failure, IRIRequestError):
            raise self.failure  # A confirmed rejection creates no job.
        self.records.append(
            {"id": f"{122 + len(self.submissions)}.polaris", "job_spec": spec,
             "status": {"state": "queued"}}
        )
        if self.failure:
            raise self.failure
        return self.records[-1]

    def jobs(self, resource, **kwargs):
        if kwargs.get("offset", 0) or (self.finished and not kwargs.get("historical")):
            return []
        account = (kwargs.get("filters") or {}).get("accountingId")
        records = [
            record for record in self.records
            if account is None or record.get("job_spec", {}).get("attributes", {}).get("account") == account
        ]
        if kwargs.get("include_spec"):
            return records
        return [{k: v for k, v in record.items() if k != "job_spec"} for record in records]

    def status(self, resource, job_id, historical=False):
        if self.finished and not historical:
            raise IRIRequestError(404)
        return {
            "id": job_id,
            "status": {"state": "completed" if self.finished else "queued"},
        }

    def cancel(self, resource, job_id):
        self.cancelled.append((resource, job_id))

    def inspect(self, resource, path, **kwargs):
        self.inspections.append((resource, path, kwargs))
        return {"text": "result"}


@pytest.fixture
def batch(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("HPC run locking requires a POSIX agent host.")
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
    # Native submissions require explicit settings; these match the target defaults.
    request = BatchRequest(
        target="polaris", launch_script="launch.sh", project="project", queue="debug",
        resources=Resources(node_count=1, duration=300, filesystems="home:eagle"),
    )
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


SUBMISSION_ID = "11111111-1111-1111-1111-111111111111"


def api_error(status, code):
    import json
    import requests

    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(
        {"code": code, "message": "private-detail", "request_id": "req-1"}
    ).encode()
    response.headers["Content-Type"] = "application/json"
    response.request = requests.Request("POST", "https://transfer.api.globus.org/v0.10/transfer").prepare()
    return globus_sdk.TransferAPIError(response)


def transfer_record(root, direction):
    if direction == "stage":
        return read_json(root / "run.json")["transfer_attempts"][-1]
    return read_json(root / ".hpc-retrievals.json")[-1]


@pytest.mark.parametrize("direction", ["stage", "retrieve"])
@pytest.mark.parametrize("failure", ["timeout", "interrupt", "save_id"])
def test_uncertain_transfer_is_replayed_with_its_submission_id(
    batch, sdk_transfer, monkeypatch, direction, failure,
):
    import chemgraph.tools.hpc.service as module

    service, root, *_ = batch
    *_, submit = sdk_transfer
    if direction == "retrieve":
        stage(batch)
        submit.reset_mock()
    evidence = root / ("run.json" if direction == "stage" else ".hpc-retrievals.json")
    original = module.write_json
    if failure == "save_id":
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
        assert result["phase"] == "submit" and result["replay"] is False
        assert result["error"] == "operation_failed" and "replay" in result["message"]
        assert "private-" not in str(result)
    first = transfer_record(root, direction)
    assert first["state"] == "unknown" and first["transfer_id"] is None
    assert first["submission_id"] == SUBMISSION_ID
    if direction == "stage":
        assert service.transfer_status(str(root))["retry_safe"] is True

    # Accepted, response lost: replaying the same ID returns the original task.
    monkeypatch.setattr(module, "write_json", original)
    submit.side_effect = None
    submit.return_value = {"code": "Duplicate", "task_id": "sdk-transfer"}
    assert tool.invoke(args)["transfer_id"] == "sdk-transfer"
    assert [call.args[0]["submission_id"] for call in submit.call_args_list] == [SUBMISSION_ID] * 2
    assert globus_sdk.TransferClient.get_submission_id.call_count == (1 if direction == "stage" else 2)
    record = transfer_record(root, direction)
    assert record["state"] == "submitted" and record["created_at"] == first["created_at"]
    assert [replay["outcome"] for replay in record["replays"]] == ["submitted"]
    if direction == "retrieve":
        assert len(read_json(evidence)) == 1


@pytest.mark.parametrize("direction", ["stage", "retrieve"])
def test_rejected_replay_preserves_the_unknown_attempt(batch, sdk_transfer, direction):
    service, root, *_ = batch
    *_, submit = sdk_transfer
    if direction == "retrieve":
        stage(batch)
        submit.reset_mock()
    tool = create_hpc_registry({}, names=["hpc_transfer_files"], service=service).get("hpc_transfer_files")
    args = {"run_dir": str(root), "direction": direction, "target": "polaris", "files": ["launch.sh"]}
    submit.side_effect = [
        TimeoutError("lost response"),          # accepted by Globus, response lost
        api_error(403, "PermissionDenied"),     # the replay itself is refused
        {"code": "Duplicate", "task_id": "original-task"},
    ]
    tool.invoke(args)
    result = tool.invoke(args)
    assert result["replay"] is True and result["retry_safe"] is True
    assert result["code"] == "PermissionDenied" and "still unknown" in result["message"]
    assert "private-detail" not in str(result)
    record = transfer_record(root, direction)
    assert record["state"] == "unknown" and record["submission_id"] == SUBMISSION_ID
    assert "error" in record and record["error"]["phase"] == "submit"
    assert record["replays"][0]["code"] == "PermissionDenied"
    if direction == "stage":
        assert service.transfer_status(str(root))["state"] == "transfer_unknown"
    assert tool.invoke(args)["transfer_id"] == "original-task"
    assert [call.args[0]["submission_id"] for call in submit.call_args_list] == [SUBMISSION_ID] * 3
    record = transfer_record(root, direction)
    assert record["state"] == "submitted"
    assert [replay["outcome"] for replay in record["replays"]] == ["failed", "submitted"]


@pytest.mark.parametrize("direction", ["stage", "retrieve"])
def test_first_request_rejection_permits_a_fresh_submission_id(batch, sdk_transfer, monkeypatch, direction):
    service, root, *_ = batch
    *_, submit = sdk_transfer
    if direction == "retrieve":
        stage(batch)
        submit.reset_mock()
    ids = iter(["22222222-2222-2222-2222-222222222222", "33333333-3333-3333-3333-333333333333"])
    monkeypatch.setattr(globus_sdk.TransferClient, "get_submission_id", Mock(
        side_effect=lambda: {"value": next(ids)},
    ))
    tool = create_hpc_registry({}, names=["hpc_transfer_files"], service=service).get("hpc_transfer_files")
    args = {"run_dir": str(root), "direction": direction, "target": "polaris", "files": ["launch.sh"]}
    submit.side_effect = [api_error(401, "AuthenticationFailed"), {"code": "Accepted", "task_id": "new-task"}]
    result = tool.invoke(args)
    assert result["error"] == "authentication_required" and result["replay"] is False
    assert transfer_record(root, direction)["state"] == "rejected"
    assert tool.invoke(args)["transfer_id"] == "new-task"
    assert [call.args[0]["submission_id"] for call in submit.call_args_list] == [
        "22222222-2222-2222-2222-222222222222", "33333333-3333-3333-3333-333333333333",
    ]


def test_unknown_transfer_without_submission_id_is_not_replayed(batch):
    service, root, transfer, *_ = batch
    manifest = stage(batch)
    manifest.update(transfer_id=None, staging="unknown")
    manifest["transfer_attempts"][-1].update(transfer_id=None, state="unknown", submission_id=None)
    write_json(root / "run.json", manifest)
    saved = (root / "run.json").read_bytes()
    tool = create_hpc_registry({}, names=["hpc_transfer_status"], service=service).get("hpc_transfer_status")
    result = tool.invoke({"run_dir": str(root)})
    assert result["state"] == "transfer_unknown" and result["transfer_id"] is None
    assert result["retry_safe"] is False and "do not restage" in result["message"]
    for transfer_id in ("unrecorded", ""):
        assert tool.invoke({"run_dir": str(root), "transfer_id": transfer_id})["error"] == "invalid_run"
    with pytest.raises(ValueError, match="no saved submission ID"):
        stage(batch)
    assert (root / "run.json").read_bytes() == saved
    assert len(transfer.calls) == 1


def test_overlapping_unknown_retrieval_requires_the_same_request(batch, sdk_transfer):
    service, root, *_ = batch
    *_, submit = sdk_transfer
    stage(batch)
    submit.side_effect = TimeoutError("lost")
    with pytest.raises(Exception):
        service.retrieve(str(root), ["a.txt", "b.txt"])
    with pytest.raises(ValueError, match="exact retrieval"):
        service.retrieve(str(root), ["a.txt"], overwrite=True)


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
    # A new session needs only the same target configuration and run directory.
    restarted = HPCService(service.config, iri=iri, transfer_factory=lambda target: transfer)
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

    service.iri = OfflineIRIClient(transport=httpx.MockTransport(respond), headers=lambda: {})
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
        assert read_json(root / "submission.json")["attempts"][-1]["state"] == "prepared"
        assert http_request.headers["Authorization"] == "Bearer test-credential"
        calls.append(http_request)
        return httpx.Response(200, json={"id": "123.polaris"})

    headers = Mock(side_effect=RuntimeError("secret provider error"))
    service.iri = OfflineIRIClient(transport=httpx.MockTransport(respond), headers=headers)
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
    service.iri = OfflineIRIClient(headers=lambda: {})
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

    service.iri = OfflineIRIClient(transport=httpx.MockTransport(fail), headers=lambda: {})
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
    "action", ["hpc_transfer_files", "hpc_submit_job", "hpc_cancel_job", "hpc_read_file", "hpc_list_files"]
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
        "hpc_read_file": {"run_dir": str(root), "path": "result.json"},
        "hpc_list_files": {"run_dir": str(root)},
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
    before = (len(transfer.calls), len(iri.submissions), len(iri.cancelled), len(iri.inspections))
    state = graph.invoke(
        {"messages": [HumanMessage(content="Run the HPC action")]}, config
    )
    assert state["__interrupt__"]
    assert before == (len(transfer.calls), len(iri.submissions), len(iri.cancelled), len(iri.inspections))
    graph.invoke(Command(resume={"decisions": [{"type": decision}]}), config)
    after = (len(transfer.calls), len(iri.submissions), len(iri.cancelled), len(iri.inspections))
    assert (after == before) == (decision == "reject")
    if action == "hpc_submit_job" and decision == "reject":
        assert not (root / "submission.started").exists()


@pytest.mark.skipif(os.name != "posix", reason="HPC targets require POSIX host paths.")
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
    with pytest.raises(ValueError, match="configured target"):
        service.status(str(root))
    with pytest.raises(ValueError, match="configured target"):
        service.cancel(str(root), "123.polaris")
    assert not iri.cancelled


@pytest.mark.parametrize(
    "tamper",
    ["absolute_identity", "traversal_identity", "file_name", "digest", "mapping", "remote_root"],
)
def test_tampered_manifest_is_rejected_before_submission(batch, tamper):
    service, root, transfer, iri, request = batch
    stage(batch)
    path = root / "run.json"
    manifest = read_json(path)
    if tamper == "absolute_identity":
        manifest["identity"] = "/etc"
        manifest["remote_directory"] = "/etc"
    elif tamper == "traversal_identity":
        manifest["identity"] = "cg/../../outside"
    elif tamper == "file_name":
        manifest["files"]["../outside"] = manifest["files"].pop("launch.sh")
    elif tamper == "digest":
        manifest["files"]["launch.sh"] = "0" * 63
    elif tamper == "mapping":
        manifest["mapping"] = {"/elsewhere/secret": "/remote/secret"}
    else:
        manifest["target_snapshot"]["remote_root"] = "/elsewhere"
    # A rejected staging attempt is the only state that replays the saved mapping.
    manifest["staging"] = "rejected"
    write_json(path, manifest)
    for call in (
        lambda: service.status(str(root)),
        lambda: service.submit(str(root), request),
        lambda: service.inspect(str(root), "stdout.txt"),
        lambda: stage(batch),
    ):
        with pytest.raises(ValueError):
            call()
    assert len(transfer.calls) == 1
    assert not iri.submissions and not iri.inspections


def test_tampered_retrieval_history_is_rejected(batch):
    service, root, transfer, *_ = batch
    manifest = stage(batch)
    (Path(manifest["remote_directory"]) / "result.json").write_text("{}")
    record = service.retrieve(str(root), ["result.json"])
    history = read_json(root / ".hpc-retrievals.json")
    history[0]["mapping"] = {"/remote/secret": "/elsewhere/secret"}
    write_json(root / ".hpc-retrievals.json", history)
    with pytest.raises(ValueError, match="Retrieval history"):
        service.transfer_status(str(root), record["transfer_id"])
    with pytest.raises(ValueError, match="Retrieval history"):
        service.retrieve(str(root), ["other.json"])
    assert len(transfer.calls) == 2


@pytest.mark.parametrize("run_dir", ["relative", "outside", "missing"])
def test_run_directory_must_be_absolute_and_configured(batch, tmp_path, run_dir):
    service, root, transfer, *_ = batch
    path = {
        "relative": "run",
        "outside": str(tmp_path / "elsewhere/run"),
        "missing": str(root.parent / "missing"),
    }[run_dir]
    tools = create_hpc_registry({}, names=["hpc_job_status", "hpc_transfer_files"], service=service)
    assert tools.get("hpc_job_status").invoke({"run_dir": path})["error"] == "invalid_run"
    if run_dir != "missing":
        result = tools.get("hpc_transfer_files").invoke(
            {"run_dir": path, "files": ["launch.sh"], "direction": "stage", "target": "polaris"}
        )
        assert result["error"] == "invalid_run"
    assert not (tmp_path / "elsewhere").exists()
    assert not (root.parent / "missing").exists()
    assert not transfer.calls


def test_unrecorded_partial_snapshot_is_replaced(batch, monkeypatch):
    import chemgraph.tools.hpc.service as module

    service, root, transfer, *_ = batch
    original = module.write_json
    monkeypatch.setattr(module, "write_json", Mock(side_effect=OSError("disk full")))
    tool = create_hpc_registry({}, names=["hpc_transfer_files"], service=service).get("hpc_transfer_files")
    args = {"run_dir": str(root), "files": ["launch.sh"], "direction": "stage", "target": "polaris"}
    result = tool.invoke(args)
    assert result["phase"] == "prepare" and result["retry_safe"] is True
    assert (root / ".hpc-inputs/launch.sh").exists() and not (root / "run.json").exists()
    monkeypatch.setattr(module, "write_json", original)
    assert tool.invoke(args)["staging"] == "submitted"
    assert len(transfer.calls) == 1


def test_partial_snapshot_with_symlink_is_left_untouched(batch, tmp_path):
    service, root, transfer, *_ = batch
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    outside.chmod(0o400)
    (root / ".hpc-inputs").mkdir()
    (root / ".hpc-inputs/launch.sh").symlink_to(outside)
    with pytest.raises(ValueError, match="unexpected entries"):
        stage(batch)
    assert (root / ".hpc-inputs/launch.sh").is_symlink()
    assert outside.read_text() == "keep" and outside.stat().st_mode & 0o777 == 0o400
    assert not transfer.calls


def test_rejected_staging_retry_reports_validation_errors(batch):
    service, root, transfer, *_ = batch
    stage(batch)
    manifest = read_json(root / "run.json")
    manifest["staging"] = "rejected"
    write_json(root / "run.json", manifest)
    (root / "inputs/water.xyz").write_text("changed")
    tool = create_hpc_registry({}, names=["hpc_transfer_files"], service=service).get("hpc_transfer_files")
    result = tool.invoke({
        "run_dir": str(root), "direction": "stage", "target": "polaris",
        "files": ["launch.sh", "inputs/water.xyz"],
    })
    assert result["error"] == "invalid_run" and "Staged inputs changed" in result["message"]
    assert len(transfer.calls) == 1


def test_unknown_target_lists_configuration(batch):
    service, root, *_ = batch
    tools = create_hpc_registry({}, names=["hpc_list_jobs", "hpc_transfer_files"], service=service)
    result = tools.get("hpc_list_jobs").invoke({"target": "aurora"})
    assert result["error"] == "invalid_run" and "'aurora'" in result["message"]
    assert "polaris" in result["message"]
    result = tools.get("hpc_transfer_files").invoke(
        {"run_dir": str(root), "files": ["launch.sh"], "direction": "stage", "target": "aurora"}
    )
    assert "configured targets" in result["message"]


def attempts(root):
    return read_json(root / "submission.json")["attempts"]


@pytest.mark.parametrize("crash", ["record", "marker", "post", "outcome"])
@pytest.mark.parametrize("resubmission", [False, True])
def test_crash_recovery_never_resends_a_marked_attempt(batch, monkeypatch, crash, resubmission):
    import chemgraph.tools.hpc.service as module

    service, root, _, iri, request = batch
    stage(batch)
    if resubmission:
        iri.failure = IRIRequestError(422)
        service.submit(str(root), request)
        iri.failure = None
    before = attempts(root) if resubmission else []
    original_write, original_mark = module.write_json, module.mark_started
    writes = []

    def write(path, value):
        writes.append(path)
        if path.name == "submission.json" and (
            (crash == "record" and len(writes) == 1) or (crash == "outcome" and len(writes) == 2)
        ):
            raise SystemExit("crash")
        original_write(path, value)

    monkeypatch.setattr(module, "write_json", write)
    if crash == "marker":
        monkeypatch.setattr(module, "mark_started", Mock(side_effect=SystemExit("crash")))
    if crash == "post":
        iri.failure = SystemExit("crash")
    with pytest.raises(SystemExit):
        service.submit(str(root), request, resubmit=resubmission)
    monkeypatch.setattr(module, "write_json", original_write)
    monkeypatch.setattr(module, "mark_started", original_mark)
    iri.failure = None
    sent = len(iri.submissions)
    result = service.submit(str(root), request, resubmit=resubmission)
    if crash in ("record", "marker"):
        # Nothing was marked, so the attempt is sent exactly once now.
        assert result["state"] == "accepted" and len(iri.submissions) == sent + 1
    else:
        assert result["state"] == "submission_unknown" and len(iri.submissions) == sent
        assert service.status(str(root))["job_id"] == iri.records[-1]["id"]
    saved = attempts(root)
    assert saved[: len(before)] == before  # Earlier evidence is never lost.
    assert len(saved) == len(before) + 1
    assert (root / "submission.started").read_text() == saved[-1]["id"]
    assert service.submit(str(root), request)["state"] == "accepted"
    assert len(iri.submissions) == int(resubmission) + 1


def test_authentication_rejection_requires_explicit_resubmission(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = IRIRequestError(401)
    result = service.submit(str(root), request)
    assert result["state"] == "rejected" and result["retry_safe"] is True
    assert result["error"]["authentication_required"] is True
    assert "python -m chemgraph.tools.alcf_iri_core login" in result["message"]
    iri.failure = None
    repeated = service.submit(str(root), request)
    assert repeated["state"] == "rejected" and "resubmit=true" in repeated["message"]
    assert len(iri.submissions) == 1
    tool = create_hpc_registry({}, names=["hpc_submit_job"], service=service).get("hpc_submit_job")
    accepted = tool.invoke({"run_dir": str(root), "request": request.model_dump(), "resubmit": True})
    assert accepted["state"] == "accepted" and accepted["attempt_count"] == 2
    first, second = attempts(root)
    assert first["state"] == "rejected" and first["id"] != second["id"]
    assert second["created_at"] >= first["created_at"]
    assert len(iri.submissions) == 2


def test_rejected_request_can_be_corrected(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = IRIRequestError(400)
    service.submit(str(root), request)
    iri.failure = None
    corrected = request.model_copy(update={"queue": "prod"})
    assert service.submit(str(root), corrected, resubmit=True)["state"] == "accepted"
    assert iri.submissions[-1][1]["attributes"]["queue_name"] == "prod"
    with pytest.raises(ValueError, match="changed"):
        service.submit(str(root), request)


def test_resubmission_rechecks_scheduler_history(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = IRIRequestError(403)
    service.submit(str(root), request)
    iri.failure = None
    # A job for this run appeared despite the saved rejection.
    iri.records.append({"id": "999.polaris", "job_spec": iri.submissions[0][1]})
    with pytest.raises(ValueError, match="does not rule out"):
        service.submit(str(root), request, resubmit=True)
    iri.records.clear()
    iri.jobs = Mock(return_value=[{"id": "other", "job_spec": {}}] * 100)
    with pytest.raises(ValueError, match="does not rule out"):
        service.submit(str(root), request, resubmit=True)
    assert len(iri.submissions) == 1 and len(attempts(root)) == 1


def test_later_duplicate_from_an_earlier_account_is_reported(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = IRIRequestError(403)
    service.submit(str(root), request)
    iri.failure = None
    other_project = request.model_copy(update={"project": "other-project"})
    accepted = service.submit(str(root), other_project, resubmit=True)
    status = service.status(str(root))
    assert status["job_id"] == accepted["job_id"] and status["duplicate_jobs"] == []
    # The earlier attempt's job appears later, under the original account.
    iri.records.append({"id": "777.polaris", "job_spec": iri.submissions[0][1]})
    status = service.status(str(root))
    assert status["duplicate_jobs"] == ["777.polaris"]
    assert status["duplicate_search_complete"] is True
    assert read_json(root / "job-status.json")["duplicate_jobs"] == ["777.polaris"]


def test_complete_empty_search_is_only_a_diagnostic(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = RuntimeError("lost")
    service.submit(str(root), request)
    iri.failure = None
    iri.records.clear()
    status = service.status(str(root))
    assert status["state"] == "submission_unknown" and status["retry_safe"] is False
    assert status["last_search"]["complete"] is True and status["last_search"]["matches"] == []
    result = service.submit(str(root), request, resubmit=True)
    assert result["state"] == "submission_unknown" and len(iri.submissions) == 1


def test_failed_cancellation_can_be_repeated(batch):
    service, root, _, iri, request = batch
    stage(batch)
    job_id = service.submit(str(root), request)["job_id"]
    with pytest.raises(ValueError, match="full accepted job ID"):
        service.cancel(str(root), "123")
    iri.cancel = Mock(side_effect=[IRIRequestError(503), None])
    with pytest.raises(IRIRequestError):
        service.cancel(str(root), job_id)
    assert read_json(root / "submission.json")["cancellations"][job_id]["state"] == "requested"
    assert "cancellation" not in attempts(root)[-1]
    assert service.cancel(str(root), job_id)["state"] == "acknowledged"
    assert iri.cancel.call_count == 2


def test_operation_polling_reuses_the_approved_request(batch):
    from chemgraph.graphs.workspace import resolve_workspace_interrupt_policy

    service, root, _, iri, _ = batch
    manifest = stage(batch)
    iri.inspect = Mock(side_effect=[
        {"task_id": "op-1"},
        {"status": "completed", "result": {"output": "data"}, "truncated": True},
    ])
    tools = create_hpc_registry(
        {}, names=["hpc_read_file", "hpc_operation_status"], service=service,
    )
    started = tools.get("hpc_read_file").invoke(
        {"run_dir": str(root), "path": "stdout.txt", "offset": 10, "size": 2048}
    )
    assert started["response"] == {"task_id": "op-1"} and "hpc_operation_status" in started["message"]
    polled = tools.get("hpc_operation_status").invoke({"run_dir": str(root), "operation_id": "op-1"})
    remote = manifest["remote_directory"] + "/stdout.txt"
    assert polled["path"] == remote and polled["operation"] == "view"
    assert polled["truncated"] is True and polled["operation_id"] == "op-1"
    assert iri.inspect.call_args.kwargs == {
        "operation": "view", "operation_id": "op-1", "offset": 10, "size": 2048,
    }
    assert tools.get("hpc_operation_status").invoke(
        {"run_dir": str(root), "operation_id": "someone-else"}
    )["error"] == "invalid_run"
    assert iri.inspect.call_count == 2
    policy = resolve_workspace_interrupt_policy()
    assert "hpc_read_file" in policy and "hpc_operation_status" not in policy


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", ".hpc-inputs/launch.sh", ""])
def test_inspection_stays_inside_the_run(batch, path):
    service, root, _, iri, _ = batch
    stage(batch)
    with pytest.raises(ValueError):
        service.inspect(str(root), path)
    assert not iri.inspections
    assert service.inspect(str(root), ".", operation="ls")["path"].endswith(read_json(root / "run.json")["identity"])


@pytest.mark.parametrize("action", ["hpc_read_file", "hpc_submit_job"])
def test_bypass_policy_reviews_only_hpc_mutations(batch, action):
    from langchain_core.messages import AIMessage, HumanMessage
    from chemgraph.graphs.deep_agent import construct_deep_agent_graph
    from chemgraph.graphs.workspace import bypass_interrupt_policy
    from tests.test_registry_middleware import CatalogModel, call

    service, root, _, iri, request = batch
    stage(batch)
    args = {
        "hpc_read_file": {"run_dir": str(root), "path": "stdout.txt"},
        "hpc_submit_job": {"run_dir": str(root), "request": request.model_dump()},
    }[action]
    registry = create_hpc_registry(service.config, names=[action], service=service)
    graph = construct_deep_agent_graph(
        CatalogModel(responses=[
            call("load_tools", names=[action]), call(action, **args), AIMessage(content="done"),
        ]),
        tool_registry=registry,
        interrupt_on=bypass_interrupt_policy(registry),
        discover_skills=False,
    )
    state = graph.invoke(
        {"messages": [HumanMessage(content="Run the HPC action")]},
        {"configurable": {"thread_id": "bypass"}},
    )
    assert bool(state.get("__interrupt__")) == (action == "hpc_submit_job")
    assert len(iri.inspections) == int(action == "hpc_read_file")
    assert not iri.submissions


def _stub_model(monkeypatch):
    from langchain_core.messages import AIMessage

    from chemgraph.models.endpoints import PreparedModel
    from tests.test_main_agent import _ScriptedChatModel

    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", lambda **kwargs: (
        _ScriptedChatModel(responses=[AIMessage(content="done")]),
        PreparedModel(endpoint_name="test", protocol="openai_compatible", client_kwargs={}),
    ))



def test_main_agent_rejects_configured_hpc_tools(monkeypatch, tmp_path):
    from chemgraph.agent import configuration
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.tools.hpc.tools import create_hpc_registry

    _stub_model(monkeypatch)
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_main_agent_graph",
        lambda *args, **kwargs: pytest.fail("must reject before graph construction"),
    )
    for registry in (create_hpc_registry({"targets": {}}), create_hpc_registry({"targets": {}}, names=["hpc_list_targets"])):
        with pytest.raises(ValueError, match="standalone deep_agent"):
            ChemGraph(workflow_type="main_agent", enable_memory=False, log_dir=str(tmp_path),
                      tool_registry=registry, approval_mode="bypass")
    # Saved main-agent catalogs are rebuilt from built-ins only.
    with pytest.raises(ValueError, match="Cannot reconstruct"):
        configuration.workspace_arguments(SimpleNamespace(registry_tool_names=("hpc_submit_job",)))



@pytest.mark.parametrize("allow", [False, True])
def test_deep_agent_bypass_keeps_hpc_mutation_reviews(monkeypatch, tmp_path, allow):
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.tools.hpc.tools import HPC_MUTATION_TOOLS, create_hpc_registry

    _stub_model(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_deep_agent_graph",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    registry = create_hpc_registry({"targets": {}, "allow_unreviewed_mutations": allow})
    ChemGraph(workflow_type="deep_agent", enable_memory=False, log_dir=str(tmp_path),
              approval_mode="bypass", deepagent_tool_registry=registry)
    expected = None if allow else {
        name: {"allowed_decisions": ["approve", "reject"]} for name in HPC_MUTATION_TOOLS
    }
    assert captured["interrupt_on"] == expected
    captured.clear()
    ChemGraph(workflow_type="deep_agent", enable_memory=False, log_dir=str(tmp_path),
              approval_mode="bypass")
    assert captured["interrupt_on"] is None  # Ordinary catalogs keep #259 semantics.


@pytest.mark.parametrize("approval_mode", ["review", "bypass"])
@pytest.mark.parametrize("workflow_type", ["deep_agent", "main_agent"])
def test_directly_attached_hpc_tools_are_rejected(monkeypatch, tmp_path, workflow_type, approval_mode):
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.tools.hpc.tools import create_hpc_tools

    _stub_model(monkeypatch)
    for name in ("construct_deep_agent_graph", "construct_main_agent_graph"):
        monkeypatch.setattr(
            f"chemgraph.agent.llm_agent.{name}",
            lambda *args, **kwargs: pytest.fail("must reject before graph construction"),
        )
    # Attached tools would carry no review tags, so bypass would leave them unreviewed.
    attached = create_hpc_tools({"targets": {}})
    mixed = {"deepagent_tool_registry": create_hpc_registry({"targets": {}})}
    for extra in ({}, mixed if workflow_type == "deep_agent" else {}):
        with pytest.raises(ValueError, match="create_hpc_registry"):
            ChemGraph(workflow_type=workflow_type, enable_memory=False, log_dir=str(tmp_path),
                      approval_mode=approval_mode, tools=[attached[3]], **extra)
    if workflow_type == "main_agent":
        with pytest.raises(ValueError, match="create_hpc_registry"):
            ChemGraph(workflow_type="main_agent", enable_memory=False, log_dir=str(tmp_path),
                      approval_mode=approval_mode,
                      subagent_options={"single_agent": {"tools": attached}})


def test_single_agent_rejects_directly_attached_hpc_tools(monkeypatch, tmp_path):
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.tools.hpc.tools import create_hpc_tools

    _stub_model(monkeypatch)
    with pytest.raises(ValueError, match="create_hpc_registry"):
        ChemGraph(workflow_type="single_agent", enable_memory=False, log_dir=str(tmp_path),
                  tools=create_hpc_tools({"targets": {}}))


def _service_with(batch, **target_updates):
    service, _, transfer, iri, _ = batch
    target = service.config.targets["polaris"].model_copy(update=target_updates)
    return HPCService(
        HPCConfig(targets={"polaris": target}), iri=iri, transfer_factory=lambda target: transfer,
    )


def test_partial_resources_keep_configured_defaults(batch):
    _, root, _, iri, _ = batch
    service = _service_with(
        batch, resources=Resources(duration=3600, filesystems="home:eagle:grand"),
    )
    stage((service, root))
    request = {"target": "polaris", "launch_script": "launch.sh", "resources": {"node_count": 2}}
    assert service.submit(str(root), request)["state"] == "accepted"
    spec = iri.submissions[-1][1]
    assert spec["resources"] == {"node_count": 2}
    assert spec["attributes"]["duration"] == 3600
    assert spec["attributes"]["custom_attributes"] == {"filesystems": "home:eagle:grand"}
    # The recorded request keeps only explicit fields, so a reload is identical.
    assert attempts(root)[-1]["intent"]["batch"] == request
    assert service.submit(str(root), BatchRequest.model_validate(request))["job_id"]
    # An explicit walltime that differs from the default is a different job.
    with pytest.raises(ValueError, match="changed"):
        service.submit(str(root), BatchRequest.model_validate(request).model_dump())
    assert len(iri.submissions) == 1


def test_edited_manifest_defaults_never_reach_the_scheduler(batch):
    service, root, _, iri, _ = batch
    stage(batch)
    manifest = read_json(root / "run.json")
    manifest["target_snapshot"].update(
        project="other", queue="prod", resources={"node_count": 64, "duration": 86400},
    )
    write_json(root / "run.json", manifest)
    service.submit(str(root), BatchRequest(target="polaris", launch_script="launch.sh"))
    spec = iri.submissions[-1][1]
    assert spec["attributes"]["account"] == "project"
    assert spec["attributes"]["queue_name"] == "debug"
    assert spec["resources"] == {"node_count": 1}
    assert spec["attributes"]["duration"] == 300


def test_accepted_attempt_keeps_its_recorded_specification(batch):
    service, root, _, iri, _ = batch
    stage(batch)
    request = BatchRequest(target="polaris", launch_script="launch.sh")
    job = service.submit(str(root), request)["job_id"]
    changed = _service_with(batch, project="renewed")
    assert changed.submit(str(root), request)["job_id"] == job
    assert attempts(root)[-1]["intent"]["jobspec"]["attributes"]["account"] == "project"
    assert changed.status(str(root))["job_id"] == job
    assert len(iri.submissions) == 1


def test_native_submission_requires_explicit_settings(batch):
    service, root, _, iri, request = batch
    stage(batch)
    tool = create_hpc_registry({}, names=["hpc_submit_job"], service=service).get("hpc_submit_job")
    implicit = {"target": "polaris", "launch_script": "launch.sh", "resources": {"node_count": 2}}
    result = tool.invoke({"run_dir": str(root), "request": implicit})
    assert result["error"] == "explicit_settings_required"
    assert result["missing"] == ["project", "queue", "resources.duration", "resources.filesystems"]
    assert result["configured_defaults"] == {
        "project": "project", "queue": "debug",
        "resources": {"node_count": 2, "duration": 300, "filesystems": "home:eagle"},
    }
    assert not iri.submissions and not (root / "submission.json").exists()
    explicit = request.model_dump()
    assert tool.invoke({"run_dir": str(root), "request": explicit})["state"] == "accepted"
    spec = iri.submissions[-1][1]
    # Every scheduler setting is taken from the reviewed arguments.
    assert spec["attributes"]["account"] == explicit["project"]
    assert spec["attributes"]["queue_name"] == explicit["queue"]
    assert spec["resources"]["node_count"] == explicit["resources"]["node_count"]
    assert spec["attributes"]["duration"] == explicit["resources"]["duration"]
    assert spec["attributes"]["custom_attributes"]["filesystems"] == explicit["resources"]["filesystems"]


def test_failed_staging_restages_the_frozen_inputs(batch):
    service, root, transfer, iri, request = batch
    stage(batch)
    transfer.state = "FAILED"
    status = service.transfer_status(str(root))
    assert status["status"] == "FAILED" and "repeat hpc_transfer_files" in status["message"]
    before = (root / "run.json").read_bytes()
    assert service.transfer_status(str(root), "transfer-1")["status"] == "FAILED"
    assert (root / "run.json").read_bytes() == before  # Status is observation only.
    stage(batch)
    manifest = read_json(root / "run.json")
    first, second = manifest["transfer_attempts"]
    assert first["state"] == "failed" and first["transfer_id"] == "transfer-1"
    assert second["state"] == "submitted" and manifest["transfer_id"] == "transfer-2"
    assert transfer.calls[0][0] == transfer.calls[1][0]  # The same frozen mapping.
    assert transfer.calls[1][1]["submission_id"] is None  # A new Globus submission ID.
    transfer.state = "SUCCEEDED"
    assert service.submit(str(root), request)["state"] == "accepted"
    # Earlier attempts stay inspectable.
    assert service.transfer_status(str(root), "transfer-1")["task_id"] == "transfer-1"


@pytest.mark.parametrize("state", ["ACTIVE", "INACTIVE", "SUCCEEDED"])
def test_unfinished_or_successful_staging_is_never_restaged(batch, state):
    service, root, transfer, *_ = batch
    stage(batch)
    transfer.state = state
    with pytest.raises(ValueError, match=f"already staged.*{state}"):
        stage(batch)
    assert len(transfer.calls) == 1


def test_failed_staging_after_submission_evidence_is_not_restaged(batch, monkeypatch):
    service, root, transfer, iri, request = batch
    stage(batch)
    monkeypatch.setattr(
        "chemgraph.tools.hpc.service.mark_started", Mock(side_effect=OSError("disk full")),
    )
    with pytest.raises(OSError):
        service.submit(str(root), request)
    assert (root / "submission.json").exists() and not (root / "submission.started").exists()
    transfer.state = "FAILED"
    assert "message" not in service.transfer_status(str(root))
    with pytest.raises(ValueError, match="already staged"):
        stage(batch)
    monkeypatch.undo()
    transfer.state = "SUCCEEDED"
    service.submit(str(root), request)
    transfer.state = "FAILED"
    with pytest.raises(ValueError, match="already attempted"):
        stage(batch)
    # Polling after submission never invalidates the fingerprinted manifest.
    service.transfer_status(str(root))
    assert service.status(str(root))["job_id"] == "123.polaris"
    assert len(transfer.calls) == 1 and len(iri.submissions) == 1


def test_verified_duplicate_jobs_can_be_cancelled(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.failure = IRIRequestError(403)
    service.submit(str(root), request)
    iri.failure = None
    accepted = service.submit(
        str(root), request.model_copy(update={"project": "other-project"}), resubmit=True,
    )["job_id"]
    iri.records.append({"id": "777.polaris", "job_spec": iri.submissions[0][1]})
    assert service.status(str(root))["duplicate_jobs"] == ["777.polaris"]
    unrelated = dict(iri.submissions[0][1], directory="/elsewhere")
    iri.records.append({"id": "888.polaris", "job_spec": unrelated})
    for job_id in ("888.polaris", "777", "999.polaris"):
        with pytest.raises(ValueError, match="verifies belongs to this run"):
            service.cancel(str(root), job_id)
    assert not iri.cancelled
    result = service.cancel(str(root), "777.polaris")
    assert result == {
        "job_id": "777.polaris", "state": "acknowledged", "at": result["at"],
        "verified_by": "scheduler_history",
    }
    assert service.cancel(str(root), accepted)["verified_by"] == "accepted_attempt"
    evidence = read_json(root / "submission.json")
    assert set(evidence["cancellations"]) == {"777.polaris", accepted}
    assert all("cancellation" not in attempt for attempt in evidence["attempts"])
    assert [job for _, job in iri.cancelled] == ["777.polaris", accepted]
    assert len(iri.submissions) == 2


def _submit_through_operation(batch):
    service, root, _, iri, request = batch
    stage(batch)
    iri.submit = Mock(return_value={"task_id": "operation"})
    assert service.submit(str(root), request)["state"] == "submission_unknown"
    return attempts(root)[-1]["intent"]["jobspec"]


@pytest.mark.parametrize("operation", [IRIRequestError(404), {"id": "operation", "status": "failed"}])
def test_missing_or_failed_operation_reconciles_from_scheduler_history(batch, operation):
    service, root, _, iri, request = batch
    spec = _submit_through_operation(batch)
    iri.task = Mock(side_effect=[operation])
    iri.records.append({"id": "321.polaris", "job_spec": spec})
    status = service.status(str(root))
    assert status["job_id"] == "321.polaris"
    current = attempts(root)[-1]
    assert current["state"] == "accepted" and current["operation_diagnostic"]["at"]
    iri.submit.assert_called_once()


@pytest.mark.parametrize("operation", [IRIRequestError(404), {"id": "operation", "status": "failed"}])
def test_missing_or_failed_operation_never_permits_resubmission(batch, operation):
    service, root, _, iri, request = batch
    _submit_through_operation(batch)
    iri.task = Mock(side_effect=[operation, operation])
    status = service.status(str(root))
    assert status["state"] == "submission_unknown" and status["retry_safe"] is False
    assert status["last_search"]["matches"] == []
    for resubmit in (False, True):
        result = service.submit(str(root), request, resubmit=resubmit)
        assert result["state"] == "submission_unknown"
    iri.submit.assert_called_once()


def test_other_operation_errors_still_surface(batch):
    service, root, _, iri, _ = batch
    _submit_through_operation(batch)
    iri.task = Mock(side_effect=IRIRequestError(503))
    with pytest.raises(IRIRequestError):
        service.status(str(root))
    assert attempts(root)[-1]["state"] == "submission_unknown"
