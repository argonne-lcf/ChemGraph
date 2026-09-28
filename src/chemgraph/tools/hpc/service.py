"""One immutable input set and one scheduler submission per run directory."""

from pathlib import Path, PurePosixPath
import shutil
import uuid

from chemgraph.execution.globus_transfer import GlobusTransferManager
from chemgraph.tools.alcf_iri_core import IRIClient, IRIRequestError
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
            if (root / "run.json").exists() or (root / "submission.started").exists():
                raise ValueError(
                    "Staging was already attempted. Use a fresh run directory."
                )
            names = [relative_path(name) for name in files]
            if not names or len(set(names)) != len(names):
                raise ValueError("Select a nonempty set of unique input files.")
            sources = {name: _input_path(root, name) for name in names}
            snapshot = root / ".hpc-inputs"
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
            identity = "cg" + uuid.uuid4().hex
            remote = str(PurePosixPath(target.remote_root) / identity)
            # Resolve and freeze resources before any remote mutation.
            target = target.model_copy(
                update={
                    "compute_resource": self.iri.resolve_resource(
                        target.compute_resource
                    ),
                    "storage_resource": self.iri.resolve_resource(
                        target.storage_resource
                    ),
                }
            )
            mapping = {
                target.local_path(snapshot / name): target.collection_path(
                    remote + "/" + name
                )
                for name in names
            }
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
            write_json(root / "run.json", manifest)
            transfer_id = self.manager(target).transfer_mapping(mapping, label=identity)
            manifest.update(transfer_id=transfer_id, staging="submitted")
            write_json(root / "run.json", manifest)
            return manifest

    def transfer_status(self, run_dir, transfer_id=None):
        with locked_run(run_dir) as root:
            manifest, target = self._load(root)
            transfer_id = transfer_id or manifest["transfer_id"]
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
                path.parent.mkdir(parents=True, exist_ok=True)
                mapping[
                    target.collection_path(manifest["remote_directory"] + "/" + name)
                ] = target.local_path(path)
            history_path = root / ".hpc-retrievals.json"
            history = read_json(history_path) if history_path.exists() else []
            # Reserve destinations even while a transfer is asynchronous or unknown.
            if not overwrite and any(
                set(mapping.values()) & set(item["mapping"].values())
                for item in history
            ):
                raise ValueError("A retrieval already targets these paths.")
            record = {"mapping": mapping, "transfer_id": None, "created_at": now()}
            history.append(record)
            write_json(history_path, history)
            record["transfer_id"] = self.manager(target).transfer_mapping(
                mapping, reverse=True, label=manifest["identity"] + " results"
            )
            write_json(history_path, history)
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
            "environment": {"CHEMGRAPH_LOG_DIR": remote},
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
                if evidence["intent"] != intent:
                    raise ValueError(
                        "Run specification changed; use a fresh run directory."
                    )
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
            evidence = {"state": "prepared", "intent": intent, "created_at": now()}
            write_json(evidence_path, evidence)
            mark_started(root)
            try:
                response = self.iri.submit(target.compute_resource, spec)
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
