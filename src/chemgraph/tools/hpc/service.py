"""One immutable input set and one scheduler submission per run directory."""

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import shutil
import uuid

from chemgraph.execution.globus_transfer import (
    GlobusTransferManager, TransferAuthenticationRequired, transfer_error_details,
)
from chemgraph.tools.alcf_iri_core import IRIAuthenticationRequired, IRIClient, IRIRequestError
from chemgraph.tools.hpc.models import BatchRequest, HPCConfig, HPCTarget, relative_path
from chemgraph.tools.hpc.store import (
    checksum,
    fingerprint,
    locked_run,
    mark_started,
    now,
    read_json,
    write_json,
)


_RESERVED = {
    "run.json",
    "submission.started",
    "submission.json",
    "job-status.json",
    "retrieved",
}


class TransferOperationError(RuntimeError):
    """Safe tool metadata for the transfer preparation/submission boundary."""

    def __init__(self, exc, phase):
        diagnostics = transfer_error_details(exc)
        self.rejected = (diagnostics.get("http_status"), diagnostics.get("code")) in {
            (403, "ConsentRequired"), (403, "PermissionDenied"),
            (403, "EndpointPermissionDenied"), (400, "BadRequest"),
            (400, "ClientError.BadRequest"),
        }
        authentication = phase == "prepare" and isinstance(
            exc, (TransferAuthenticationRequired, IRIAuthenticationRequired)
        )
        message = (
            "Transfer preparation failed; correct the problem and retry in the same run directory."
            if phase == "prepare" else
            "Transfer submission outcome is unknown. Inspect saved evidence and Globus task history; "
            "do not retry the transfer or delete recorded attempts."
        )
        if authentication:
            message = f"{exc} Run the login command in a terminal. {message}"
        elif self.rejected:
            reason = {
                "ConsentRequired": "Authenticate in a terminal with the required collection data_access scopes.",
                "PermissionDenied": "Verify the authenticated identity has access to both collections.",
                "EndpointPermissionDenied": "Verify the collection permits access to the configured paths.",
            }.get(diagnostics["code"], "Correct the transfer request configuration.")
            message = (
                f"Globus rejected the request. {reason} "
                "Retry explicitly in the same run directory after correcting the problem; preserve all attempts."
            )
        self.details = {
            "error": "authentication_required" if authentication else "operation_failed",
            "type": type(exc).__name__,
            "phase": phase,
            "retry_safe": phase == "prepare" or self.rejected,
            "message": message,
            **diagnostics,
        }
        super().__init__(message)


@contextmanager
def _transfer_phase(phase):
    try:
        yield
    except Exception as exc:
        raise TransferOperationError(exc, phase) from None


def _input_path(root, name):
    name = relative_path(name)
    if PurePosixPath(name).parts[0] in _RESERVED:
        raise ValueError("Input path conflicts with run evidence.")
    path = root / name
    for part in [path, *path.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Symlink inputs are unsupported.")
    if not path.is_file():
        raise ValueError(f"Missing input file: {name}")
    return path


class HPCService:
    def __init__(self, config, *, iri=None, transfer_factory=None):
        self.config = HPCConfig.model_validate(config).model_copy(deep=True)
        self.iri = iri or IRIClient()
        self.transfer_factory = transfer_factory or self._transfer_manager
        self._managers = {}

    @staticmethod
    def _transfer_manager(target):
        return GlobusTransferManager(
            target.local_collection,
            target.remote_collection,
            target.remote_collection_root,
        )

    def manager(self, target):
        key = fingerprint(target.model_dump())
        if key not in self._managers:
            self._managers[key] = self.transfer_factory(target)
        return self._managers[key]

    def list_targets(self):
        return self.config.model_dump()

    def _load(self, root):
        manifest = read_json(root / "run.json")
        if manifest.get("version") != 1:
            raise ValueError("Unsupported run manifest version.")
        if manifest["local_directory"] != str(root):
            raise ValueError("Run directory moved; restore its original location.")
        target = HPCTarget.model_validate(manifest["target_snapshot"])
        expected_remote = str(PurePosixPath(target.remote_root) / manifest["identity"])
        if manifest["remote_directory"] != expected_remote:
            raise ValueError("Run remote directory differs from its saved identity.")
        evidence_path = root / "submission.json"
        if evidence_path.exists():
            evidence = read_json(evidence_path)
            if evidence.get("intent", {}).get("manifest_hash") != fingerprint(manifest):
                raise ValueError("Run manifest changed after submission preparation.")
        return manifest, target

    def stage(self, run_dir, target_name, files):
        target = self.config.targets[target_name]
        with locked_run(run_dir) as root:
            root.relative_to(Path(target.local_root).resolve())
            if (root / "submission.started").exists():
                raise ValueError(
                    "Staging was already attempted; inspect its status, do not retry in a fresh directory."
                )
            names = [relative_path(name) for name in files]
            if not names or len(set(names)) != len(names):
                raise ValueError("Select a nonempty set of unique input files.")
            if (root / "run.json").exists():
                manifest, saved_target = self._load(root)
                if manifest.get("staging") != "rejected" or (root / "submission.json").exists():
                    raise ValueError("Staging was already attempted; inspect its status, do not retry in a fresh directory.")
                if target_name != manifest["target"] or set(names) != set(manifest["files"]):
                    raise ValueError("Retry must use the same target and input files.")
                with _transfer_phase("prepare"):
                    resolved = target.model_copy(update={
                        "compute_resource": self.iri.resolve_resource(target.compute_resource),
                        "storage_resource": self.iri.resolve_resource(target.storage_resource),
                    })
                    if resolved != saved_target:
                        raise ValueError("Retry must preserve the saved target configuration.")
                    self._check_inputs(root, manifest)
                    manager = self.manager(saved_target)
                    prepared = manager.prepare_mapping(manifest["mapping"], label=manifest["identity"])
                return self._submit_stage(root, manifest, manager, prepared)
            sources = {name: _input_path(root, name) for name in names}
            snapshot = root / ".hpc-inputs"
            if snapshot.exists():
                raise ValueError("A partial snapshot exists; use a fresh run directory.")
            identity = "cg" + uuid.uuid4().hex
            remote = str(PurePosixPath(target.remote_root) / identity)
            with _transfer_phase("prepare"):
                # Freeze resources and prepare planned snapshot paths before writing evidence.
                target = target.model_copy(
                    update={
                        "compute_resource": self.iri.resolve_resource(target.compute_resource),
                        "storage_resource": self.iri.resolve_resource(target.storage_resource),
                    }
                )
                mapping = {
                    target.local_path(snapshot / name): target.collection_path(remote + "/" + name)
                    for name in names
                }
                manager = self.manager(target)
                prepared = manager.prepare_mapping(mapping, label=identity)
            snapshot.mkdir()  # A partial snapshot also requires a fresh run.
            identities = {}
            for name, source in sources.items():
                destination = snapshot / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                before = checksum(source)
                shutil.copyfile(source, destination)
                if checksum(destination) != before or checksum(source) != before:
                    raise ValueError("Inputs changed during staging; use a fresh run.")
                destination.chmod(0o400)
                identities[name] = before
            manifest = {
                "version": 1,
                "identity": identity,
                "target": target_name,
                "target_snapshot": target.model_dump(),
                "local_directory": str(root),
                "remote_directory": remote,
                "files": identities,
                "mapping": mapping,
                "created_at": now(),
                "transfer_id": None,
                "staging": "unknown",
            }
            return self._submit_stage(root, manifest, manager, prepared)

    @staticmethod
    def _submit_transfer(manager, prepared, record, save):
        record.update(
            submission_id=getattr(prepared, "submission_id", None),
            transfer_id=None, state="unknown", created_at=now(),
        )
        with _transfer_phase("prepare"):
            save()
        try:
            transfer_id = manager.submit_prepared(prepared)
        except Exception as exc:
            error = TransferOperationError(exc, "submit")
            record["error"] = error.details
            if error.rejected:
                record["state"] = "rejected"
            # A failure saving the rejection must leave the attempt unknown.
            with _transfer_phase("submit"):
                save()
            raise error from None
        with _transfer_phase("submit"):
            record.update(transfer_id=transfer_id, state="submitted")
            save()

    def _submit_stage(self, root, manifest, manager, prepared):
        record = {}
        manifest.setdefault("transfer_attempts", []).append(record)

        def save():
            manifest.update(transfer_id=record["transfer_id"], staging=record["state"])
            write_json(root / "run.json", manifest)

        self._submit_transfer(manager, prepared, record, save)
        return manifest

    def transfer_status(self, run_dir, transfer_id=None):
        with locked_run(run_dir) as root:
            manifest, target = self._load(root)
            if transfer_id is None and not manifest["transfer_id"]:
                if manifest.get("staging") == "rejected":
                    return {
                        **manifest["transfer_attempts"][-1]["error"],
                        "state": "transfer_rejected", "transfer_id": None,
                    }
                return {
                    "state": "transfer_unknown",
                    "transfer_id": None,
                    "retry_safe": False,
                    "message": "A staging attempt is recorded without a task ID. Inspect Globus task history; "
                    "do not restage this run, retry in a new directory, or delete recorded attempts.",
                }
            transfer_id = manifest["transfer_id"] if transfer_id is None else transfer_id
            known = {manifest["transfer_id"]}
            history = root / ".hpc-retrievals.json"
            if history.exists():
                known.update(item.get("transfer_id") for item in read_json(history))
            if not transfer_id or transfer_id not in known:
                raise ValueError(
                    "No matching recorded transfer ID; do not restage this run."
                )
            return self.manager(target).check_transfer_status(transfer_id)

    def retrieve(self, run_dir, files, *, overwrite=False):
        with locked_run(run_dir) as root:
            manifest, target = self._load(root)
            names = [relative_path(name) for name in files]
            if not names or len(set(names)) != len(names):
                raise ValueError("Select unique output files.")
            destination = root / "retrieved"
            mapping = {}
            for name in names:
                path = destination / name
                if any(
                    part.is_symlink() for part in [path, *path.parents] if part != root
                ):
                    raise ValueError("Symlink retrieval destinations are unsupported.")
                if path.exists() and not overwrite:
                    raise ValueError(
                        "Retrieval destination exists; explicitly request overwrite."
                    )
                mapping[
                    target.collection_path(manifest["remote_directory"] + "/" + name)
                ] = target.local_path(path)
            history_path = root / ".hpc-retrievals.json"
            history = read_json(history_path) if history_path.exists() else []
            for item in history:
                if not set(mapping.values()) & set(item["mapping"].values()):
                    continue
                if item.get("state") == "rejected":
                    continue
                if not item.get("transfer_id"):
                    raise ValueError("A retrieval targeting these paths has an unknown outcome; do not retry.")
                if not overwrite:
                    raise ValueError("A retrieval already targets these paths.")
                if self.manager(target).check_transfer_status(item["transfer_id"])["status"] not in {"SUCCEEDED", "FAILED"}:
                    raise ValueError("A retrieval targeting these paths is still active; wait before overwriting.")
            with _transfer_phase("prepare"):
                manager = self.manager(target)
                prepared = manager.prepare_mapping(
                    mapping, reverse=True, label=manifest["identity"] + " results"
                )
            for name in names:
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
            record = {"mapping": mapping}
            history.append(record)
            self._submit_transfer(
                manager, prepared, record, lambda: write_json(history_path, history),
            )
            return record

    @staticmethod
    def _jobspec(manifest, target, request):
        remote = manifest["remote_directory"]
        resources = request.resources or target.resources
        return {
            "executable": "/bin/bash",
            "arguments": [remote + "/" + request.launch_script, *request.arguments],
            "directory": remote,
            "name": manifest["identity"][:15],
            "stdout_path": remote + "/" + request.stdout,
            "stderr_path": remote + "/" + request.stderr,
            "environment": {},
            "resources": {"node_count": resources.node_count},
            "attributes": {
                "duration": resources.duration,
                "queue_name": request.queue or target.queue,
                "account": request.project or target.project,
                "custom_attributes": {"filesystems": resources.filesystems},
            },
        }

    def submit(self, run_dir, request):
        request = BatchRequest.model_validate(request)
        with locked_run(run_dir) as root:
            manifest, target = self._load(root)
            if request.target != manifest["target"]:
                raise ValueError("Request target differs from this run.")
            spec = self._jobspec(manifest, target, request)
            intent = {
                "batch": request.model_dump(),
                "jobspec": spec,
                "manifest_hash": fingerprint(manifest),
            }
            evidence_path = root / "submission.json"
            evidence = read_json(evidence_path) if evidence_path.exists() else None
            if evidence:
                # Older runs injected this one application-specific variable.
                # Accept only the exact former specification, not arbitrary drift.
                legacy_spec = {
                    **spec,
                    "environment": {"CHEMGRAPH_LOG_DIR": manifest["remote_directory"]},
                }
                legacy_intent = {**intent, "jobspec": legacy_spec}
                if evidence["intent"] not in (intent, legacy_intent):
                    raise ValueError(
                        "Run specification changed; use a fresh run directory."
                    )
                intent = evidence["intent"]
                spec = intent["jobspec"]
                # Check input identity even for an accepted repeated request.
                self._check_inputs(root, manifest)
                if evidence["state"] != "prepared":
                    return evidence
            if (root / "submission.started").exists():
                return {
                    "state": "submission_unknown",
                    "message": "Inspect this run; never resubmit.",
                }
            self._check_inputs(root, manifest)
            if request.launch_script not in manifest["files"]:
                raise ValueError("Launch script must be in the staged manifest.")
            if any(
                name in manifest["files"]
                for name in [request.stdout, request.stderr, *request.expected_outputs]
            ):
                raise ValueError("Outputs must not overwrite staged inputs.")
            if "/" in request.stdout or "/" in request.stderr:
                raise ValueError(
                    "Scheduler stdout/stderr must be filenames in the run root."
                )
            transfer_id = manifest["transfer_id"]
            if (
                not transfer_id
                or self.manager(target).check_transfer_status(transfer_id)["status"]
                != "SUCCEEDED"
            ):
                raise ValueError(
                    "Input transfer must complete successfully before submission."
                )
            prepared = self.iri.prepare_submission(target.compute_resource, spec)
            if evidence is None:
                evidence = {"state": "prepared", "intent": intent, "created_at": now()}
            write_json(evidence_path, evidence)
            mark_started(root)
            try:
                response = self.iri.submit_prepared(prepared)
                if isinstance(response, dict) and response.get("id"):
                    evidence.update(state="accepted", job_id=str(response["id"]))
                else:
                    evidence.update(state="submission_unknown")
                    if isinstance(response, dict) and response.get("task_id"):
                        evidence["operation_id"] = str(response["task_id"])
            except Exception as exc:
                definitive = isinstance(exc, IRIRequestError) and exc.status_code in (
                    400,
                    401,
                    403,
                    404,
                    422,
                )
                evidence.update(
                    state="rejected" if definitive else "submission_unknown",
                    error={"type": type(exc).__name__},
                )
                if isinstance(exc, IRIRequestError):
                    evidence["error"]["http_status"] = exc.status_code
            evidence["updated_at"] = now()
            write_json(evidence_path, evidence)
            return evidence

    @staticmethod
    def _check_inputs(root, manifest):
        for name, digest in manifest["files"].items():
            if (
                checksum(_input_path(root, name)) != digest
                or checksum(root / ".hpc-inputs" / name) != digest
            ):
                raise ValueError("Staged inputs changed; use a fresh run directory.")

    def _reconcile(self, manifest, target, evidence):
        if not evidence or evidence["state"] not in ("prepared", "submission_unknown"):
            return evidence
        if evidence.get("operation_id"):
            operation = self.iri.task(evidence["operation_id"])
            result = operation.get("result") if isinstance(operation, dict) else None
            if (
                isinstance(result, dict)
                and operation.get("status") == "completed"
                and result.get("id")
            ):
                evidence.update(
                    state="accepted", job_id=str(result["id"]), reconciled_at=now()
                )
                return evidence
            if isinstance(operation, dict) and operation.get("status") in (
                "pending",
                "active",
            ):
                evidence["state"] = "submission_unknown"
                return evidence
        matches = {}
        for historical in (False, True):
            for offset in range(0, 1000, 100):
                records = self.iri.jobs(
                    target.compute_resource,
                    historical=historical,
                    include_spec=True,
                    limit=100,
                    offset=offset,
                    filters={
                        "accountingId": evidence["intent"]["jobspec"]["attributes"][
                            "account"
                        ]
                    },
                )
                if not isinstance(records, list):
                    return evidence  # Unknown response shapes never establish identity.
                for record in records:
                    spec = record.get("job_spec") or {}
                    expected = evidence["intent"]["jobspec"]
                    if (
                        record.get("id")
                        and spec.get("name") == expected["name"]
                        and spec.get("directory") == manifest["remote_directory"]
                        and spec.get("stdout_path") == expected["stdout_path"]
                        and spec.get("attributes", {}).get("account")
                        == expected["attributes"]["account"]
                    ):
                        matches[str(record["id"])] = record
                if len(records) < 100:
                    break
            else:
                return evidence  # Search bound exceeded: matching is incomplete.
        if len(matches) == 1:
            evidence.update(
                state="accepted", job_id=next(iter(matches)), reconciled_at=now()
            )
        else:
            evidence["state"] = "submission_unknown"
        return evidence

    def status(self, run_dir):
        with locked_run(run_dir) as root:
            manifest, target = self._load(root)
            path = root / "submission.json"
            if not path.exists():
                return {
                    "state": "submission_unknown"
                    if (root / "submission.started").exists()
                    else "not_submitted"
                }
            evidence = read_json(path)
            if (root / "submission.started").exists():
                evidence = self._reconcile(manifest, target, evidence)
                write_json(path, evidence)
            if evidence["state"] != "accepted":
                return evidence
            job_id = evidence["job_id"]
            try:
                record = self.iri.status(target.compute_resource, job_id)
            except IRIRequestError as exc:
                if exc.status_code != 404:
                    raise
                record = self.iri.status(
                    target.compute_resource, job_id, historical=True
                )
            status = {
                "job_id": job_id,
                "target": manifest["target"],
                "observed_at": now(),
                "scheduler": record.get("status"),
                "scientific_success": None,
            }
            write_json(root / "job-status.json", status)
            return status

    def list_jobs(self, target, **kwargs):
        resource = self.iri.resolve_resource(
            self.config.targets[target].compute_resource
        )
        return self.iri.jobs(resource, **kwargs)

    def cancel(self, run_dir, job_id):
        with locked_run(run_dir) as root:
            manifest, target = self._load(root)
            path = root / "submission.json"
            evidence = read_json(path)
            if evidence.get("job_id") != job_id or evidence["state"] != "accepted":
                raise ValueError(
                    "Cancellation must match this run's full accepted job ID."
                )
            # A cancellation request is not proof that the job has stopped.
            evidence["cancellation"] = {"state": "requested", "at": now()}
            write_json(path, evidence)
            self.iri.cancel(target.compute_resource, job_id)
            evidence["cancellation"]["state"] = "acknowledged"
            write_json(path, evidence)
            return evidence["cancellation"]

    def inspect(
        self,
        run_dir,
        path=".",
        *,
        operation="view",
        operation_id=None,
        offset=0,
        size=16384,
    ):
        with locked_run(run_dir) as root:
            manifest, target = self._load(root)
            if path != ".":
                path = relative_path(path)
            remote = str(PurePosixPath(manifest["remote_directory"]) / path)
            result = self.iri.inspect(
                target.storage_resource,
                remote,
                operation=operation,
                operation_id=operation_id,
                offset=offset,
                size=size,
            )
            return {
                "path": remote,
                "response": result,
                "message": "If task_id is returned, call again with operation_id to inspect its result.",
            }
