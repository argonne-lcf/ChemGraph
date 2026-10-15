"""One immutable input set and one scheduler submission per run directory."""

from contextlib import contextmanager
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
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
    started_attempt,
    write_json,
)


_RESERVED = {
    "run.json",
    "submission.started",
    "submission.json",
    "job-status.json",
    "retrieved",
}
_IDENTITY = re.compile(r"cg[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SUBMISSION_ID = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
# HTTP statuses that prove an IRI submission was not accepted.
_DEFINITIVE_REJECTIONS = (400, 401, 403, 404, 422)
# Fields that place files; a run must keep the configuration it was staged with.
_TARGET_PATHS = (
    "local_collection", "remote_collection", "local_root",
    "local_collection_root", "remote_root", "remote_collection_root",
)


class TransferOperationError(RuntimeError):
    """Safe tool metadata for the transfer preparation/submission boundary.

    ``replay`` marks a repeated submission of a recorded ID. Its failure says
    nothing about the original request, which therefore stays unknown.
    """

    def __init__(self, exc, phase, *, replay=False):
        diagnostics = transfer_error_details(exc)
        self.rejected = not replay and (diagnostics.get("http_status"), diagnostics.get("code")) in {
            (401, "AuthenticationFailed"),
            (403, "ConsentRequired"), (403, "PermissionDenied"),
            (403, "EndpointPermissionDenied"), (400, "BadRequest"),
            (400, "ClientError.BadRequest"),
        }
        # Local authentication errors carry only fixed text; SDK errors do not.
        local_authentication = isinstance(
            exc, (TransferAuthenticationRequired, IRIAuthenticationRequired)
        )
        authentication = local_authentication or diagnostics.get("http_status") == 401
        if phase == "prepare":
            message = "Transfer preparation failed; correct the problem and retry in the same run directory."
        elif replay:
            message = (
                "Replaying the recorded submission failed; the original outcome is still unknown. "
                "Correct the problem and repeat the same call: the saved submission ID prevents a duplicate transfer."
            )
        else:
            message = (
                "Transfer submission outcome is unknown. Repeat the same call to replay its saved "
                "submission ID; never start the transfer in another run directory."
            )
        if local_authentication:
            message = f"{exc} Run the login command in a terminal. {message}"
        elif authentication and not self.rejected:
            message = f"Globus Transfer authentication failed; run the login command in a terminal. {message}"
        elif self.rejected:
            reason = {
                "ConsentRequired": "Authenticate in a terminal with the required collection data_access scopes.",
                "PermissionDenied": "Verify the authenticated identity has access to both collections.",
                "EndpointPermissionDenied": "Verify the collection permits access to the configured paths.",
                "AuthenticationFailed": "Authenticate Globus Transfer in a terminal.",
            }.get(diagnostics["code"], "Correct the transfer request configuration.")
            message = (
                f"Globus rejected the request. {reason} "
                "Retry explicitly in the same run directory after correcting the problem; preserve all attempts."
            )
        self.details = {
            "error": "authentication_required" if authentication else "operation_failed",
            "type": type(exc).__name__,
            "phase": phase,
            "replay": replay,
            # Repeating the call is safe: it prepares again, replays the
            # recorded ID, or starts a fresh attempt after a confirmed rejection.
            "retry_safe": True,
            "message": message,
            **diagnostics,
        }
        super().__init__(message)


@contextmanager
def _transfer_phase(phase, *, replay=False):
    try:
        yield
    except Exception as exc:
        raise TransferOperationError(exc, phase, replay=replay) from None


def _input_name(name):
    if not isinstance(name, str) or relative_path(name) != name:
        raise ValueError("Input names must be normalized relative paths.")
    if PurePosixPath(name).parts[0] in _RESERVED:
        raise ValueError("Input path conflicts with run evidence.")
    return name


def _stage_mapping(target, root, identity, names):
    """Rebuild the only staging mapping a run identity may replay."""
    remote = str(PurePosixPath(target.remote_root) / identity)
    snapshot = root / ".hpc-inputs"
    return {
        target.local_path(snapshot / name): target.collection_path(remote + "/" + name)
        for name in names
    }


def _retrieval_mapping(target, root, remote, names):
    destination = root / "retrieved"
    return {
        target.collection_path(remote + "/" + name): target.local_path(destination / name)
        for name in names
    }


def _remove_partial_snapshot(snapshot):
    """Remove an unrecorded snapshot only when it holds plain files and directories."""
    entries = [snapshot]
    for directory, dirs, files in os.walk(snapshot):
        entries.extend(Path(directory) / name for name in dirs + files)
    modes = [os.lstat(entry).st_mode for entry in entries]
    if any(
        stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode))
        for mode in modes
    ):
        raise ValueError("A partial snapshot contains unexpected entries; inspect it manually.")
    for entry, mode in zip(entries, modes):
        entry.chmod(0o700 if stat.S_ISDIR(mode) else 0o600)
    shutil.rmtree(snapshot)


def _replayable(record):
    submission_id = record.get("submission_id") if isinstance(record, dict) else None
    if (
        not isinstance(submission_id, str)
        or not _SUBMISSION_ID.fullmatch(submission_id)
        or record.get("state") != "unknown"
    ):
        raise ValueError(
            "An earlier transfer has an unknown outcome and no saved submission ID; "
            "inspect Globus task history before any further transfer."
        )
    return record


def _input_path(root, name):
    name = _input_name(relative_path(name))
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

    def _target(self, name):
        try:
            return self.config.targets[name]
        except KeyError:
            raise ValueError(
                f"Unknown HPC target {name!r}; configured targets: {sorted(self.config.targets)}."
            ) from None

    def _run(self, run_dir, *, create=False):
        roots = [Path(target.local_root).resolve() for target in self.config.targets.values()]
        return locked_run(run_dir, roots=roots, create=create)

    def _resolved(self, target):
        return target.model_copy(update={
            "compute_resource": self.iri.resolve_resource(target.compute_resource),
            "storage_resource": self.iri.resolve_resource(target.storage_resource),
        })

    def _check_target(self, name, snapshot):
        """The configured target, not the editable manifest, authorizes locations."""
        configured = self._target(name)
        resolved = self._resolved(configured)
        for field in (*_TARGET_PATHS, "compute_resource", "storage_resource"):
            if getattr(resolved, field) != getattr(snapshot, field):
                raise ValueError(
                    f"Run manifest does not match configured target {name!r} ({field}); "
                    "restore the target configuration used for this run."
                )

    def _load(self, root):
        manifest = read_json(root / "run.json")
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise ValueError("Unsupported run manifest version.")
        if manifest.get("local_directory") != str(root):
            raise ValueError("Run directory moved; restore its original location.")
        identity = manifest.get("identity")
        if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
            raise ValueError("Run manifest identity is invalid.")
        target = HPCTarget.model_validate(manifest.get("target_snapshot"))
        self._check_target(manifest.get("target"), target)
        if not root.is_relative_to(Path(target.local_root).resolve()):
            raise ValueError("Run directory is outside its target local_root.")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files or not all(
            isinstance(digest, str) and _DIGEST.fullmatch(digest) for digest in files.values()
        ):
            raise ValueError("Run manifest input identities are invalid.")
        for name in files:
            _input_name(name)
        expected_remote = str(PurePosixPath(target.remote_root) / identity)
        if manifest.get("remote_directory") != expected_remote:
            raise ValueError("Run remote directory differs from its saved identity.")
        if manifest.get("mapping") != _stage_mapping(target, root, identity, files):
            raise ValueError("Run transfer mapping differs from its saved identity.")
        evidence_path = root / "submission.json"
        if evidence_path.exists():
            evidence = read_json(evidence_path)
            if evidence.get("manifest_hash") != fingerprint(manifest):
                raise ValueError("Run manifest changed after submission preparation.")
        return manifest, target

    def _retrievals(self, root, manifest, target):
        """Load retrieval history, accepting only mappings this run can produce."""
        path = root / ".hpc-retrievals.json"
        history = read_json(path) if path.exists() else []
        if not isinstance(history, list):
            raise ValueError("Retrieval history is invalid.")
        for item in history:
            names = item.get("files") if isinstance(item, dict) else None
            if not isinstance(names, list) or item.get("mapping") != _retrieval_mapping(
                target, root, manifest["remote_directory"],
                [relative_path(name) for name in names],
            ):
                raise ValueError("Retrieval history differs from this run's identity.")
        return path, history

    def stage(self, run_dir, target_name, files):
        target = self._target(target_name)
        with self._run(run_dir, create=True) as root:
            if not root.is_relative_to(Path(target.local_root).resolve()):
                raise ValueError(f"Run directory is outside target {target_name!r} local_root.")
            if (root / "submission.started").exists():
                raise ValueError(
                    "Staging was already attempted; inspect its status, do not retry in a fresh directory."
                )
            names = [relative_path(name) for name in files]
            if not names or len(set(names)) != len(names):
                raise ValueError("Select a nonempty set of unique input files.")
            if (root / "run.json").exists():
                manifest, saved_target = self._load(root)
                staging = manifest.get("staging")
                if (
                    staging not in ("rejected", "unknown", "submitted")
                    or (root / "submission.json").exists()
                ):
                    raise ValueError(
                        "Inputs were already staged; check hpc_transfer_status. "
                        "Changed inputs need a fresh run directory."
                    )
                if target_name != manifest["target"] or set(names) != set(manifest["files"]):
                    raise ValueError("Retry must use the same target and input files.")
                if self._resolved(target) != saved_target:
                    raise ValueError("Retry must preserve the saved target configuration.")
                self._check_inputs(root, manifest)
                if staging == "submitted":
                    # Only a transfer Globus reports as terminally FAILED may be
                    # followed by a new attempt; an active one may still finish.
                    observed = self.manager(saved_target).check_transfer_status(
                        manifest["transfer_id"]
                    )["status"]
                    if observed != "FAILED":
                        raise ValueError(
                            f"Inputs were already staged (transfer status {observed}); check "
                            "hpc_transfer_status. Only a FAILED staging transfer may be retried."
                        )
                    manifest["transfer_attempts"][-1].update(state="failed", failed_at=now())
                # An unknown attempt is replayed with its own ID; only a confirmed
                # rejection of an attempt's first request permits a new ID.
                attempt = _replayable(manifest["transfer_attempts"][-1]) if staging == "unknown" else None
                with _transfer_phase("prepare"):
                    manager = self.manager(saved_target)
                    prepared = manager.prepare_mapping(
                        manifest["mapping"], label=manifest["identity"],
                        submission_id=attempt and attempt["submission_id"],
                    )
                return self._submit_stage(root, manifest, manager, prepared, record=attempt)
            sources = {name: _input_path(root, name) for name in names}
            snapshot = root / ".hpc-inputs"
            if snapshot.exists() or snapshot.is_symlink():
                # Without run.json no transfer was submitted: evidence is saved first.
                _remove_partial_snapshot(snapshot)
            identity = "cg" + uuid.uuid4().hex
            remote = str(PurePosixPath(target.remote_root) / identity)
            # Freeze resources and plan snapshot paths before writing evidence.
            target = self._resolved(target)
            mapping = _stage_mapping(target, root, identity, names)
            with _transfer_phase("prepare"):
                manager = self.manager(target)
                prepared = manager.prepare_mapping(mapping, label=identity)
            snapshot.mkdir()
            identities = {}
            for name, source in sources.items():
                destination = snapshot / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                before = checksum(source)
                shutil.copyfile(source, destination)
                if checksum(destination) != before or checksum(source) != before:
                    raise ValueError("Inputs changed during staging; correct them and retry.")
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
    def _submit_transfer(manager, prepared, record, save, *, replay=False):
        """Send a new attempt, or replay an unknown one without downgrading it.

        Only the first request of an attempt may record a rejection. A replay
        keeps the original ID, timestamp and unknown state on failure and logs
        its own diagnostics separately.
        """
        if not replay:
            record.update(
                submission_id=getattr(prepared, "submission_id", None),
                transfer_id=None, state="unknown", created_at=now(),
            )
            with _transfer_phase("prepare"):
                save()
        try:
            transfer_id = manager.submit_prepared(prepared)
        except Exception as exc:
            error = TransferOperationError(exc, "submit", replay=replay)
            if replay:
                record.setdefault("replays", []).append(
                    {"at": now(), "outcome": "failed", **transfer_error_details(exc),
                     "type": type(exc).__name__}
                )
            else:
                record["error"] = error.details
                if error.rejected:
                    record["state"] = "rejected"
            # A failure saving the rejection must leave the attempt unknown.
            with _transfer_phase("submit", replay=replay):
                save()
            raise error from None
        with _transfer_phase("submit", replay=replay):
            if replay:
                record.setdefault("replays", []).append({"at": now(), "outcome": "submitted"})
            record.update(transfer_id=transfer_id, state="submitted")
            save()

    def _submit_stage(self, root, manifest, manager, prepared, *, record=None):
        replay = record is not None
        if not replay:
            record = {}
            manifest.setdefault("transfer_attempts", []).append(record)

        def save():
            manifest.update(transfer_id=record["transfer_id"], staging=record["state"])
            write_json(root / "run.json", manifest)

        self._submit_transfer(manager, prepared, record, save, replay=replay)
        return manifest

    def transfer_status(self, run_dir, transfer_id=None):
        with self._run(run_dir) as root:
            manifest, target = self._load(root)
            if transfer_id is None and not manifest["transfer_id"]:
                if manifest.get("staging") == "rejected":
                    return {
                        **manifest["transfer_attempts"][-1]["error"],
                        "state": "transfer_rejected", "transfer_id": None,
                    }
                replay = bool(manifest["transfer_attempts"][-1].get("submission_id"))
                return {
                    "state": "transfer_unknown",
                    "transfer_id": None,
                    "retry_safe": replay,
                    "message": (
                        "A staging attempt is recorded without a task ID. Repeat hpc_transfer_files with "
                        "the same target and files to replay its saved submission ID; Globus returns the "
                        "original task if it was accepted. Never restage in a new directory."
                        if replay else
                        "A staging attempt is recorded without a task ID or submission ID. Inspect Globus "
                        "task history; do not restage this run, retry in a new directory, or delete recorded attempts."
                    ),
                }
            transfer_id = manifest["transfer_id"] if transfer_id is None else transfer_id
            known = {manifest["transfer_id"]}
            known.update(item.get("transfer_id") for item in manifest["transfer_attempts"])
            known.update(item.get("transfer_id") for item in self._retrievals(root, manifest, target)[1])
            if not transfer_id or transfer_id not in known:
                raise ValueError(
                    "No matching recorded transfer ID; do not restage this run."
                )
            # Observation only: run.json is never rewritten here, because
            # submission evidence fingerprints the manifest.
            status = self.manager(target).check_transfer_status(transfer_id)
            if (
                transfer_id == manifest["transfer_id"]
                and status.get("status") == "FAILED"
                and not (root / "submission.json").exists()
            ):
                status["message"] = (
                    "Staging failed. Correct the cause, then repeat hpc_transfer_files with the "
                    "same target and files in this run directory to stage the frozen inputs again."
                )
            return status

    def retrieve(self, run_dir, files, *, overwrite=False):
        with self._run(run_dir) as root:
            manifest, target = self._load(root)
            names = [relative_path(name) for name in files]
            if not names or len(set(names)) != len(names):
                raise ValueError("Select unique output files.")
            destination = root / "retrieved"
            for name in names:
                path = destination / name
                if any(
                    part.is_symlink() for part in [path, *path.parents] if part != root
                ):
                    raise ValueError("Symlink retrieval destinations are unsupported.")
            mapping = _retrieval_mapping(target, root, manifest["remote_directory"], names)
            history_path, history = self._retrievals(root, manifest, target)
            overlapping = [
                item for item in history
                if set(mapping.values()) & set(item["mapping"].values())
                and item.get("state") != "rejected"
            ]
            unknown = [item for item in overlapping if not item.get("transfer_id")]
            record = None
            if unknown:
                # Only the exact earlier request may be replayed, with its own ID.
                if len(unknown) > 1 or unknown[0]["mapping"] != mapping:
                    raise ValueError(
                        "A retrieval of overlapping paths has an unknown outcome; repeat that exact "
                        f"retrieval (files={unknown[0]['files']}) to replay it."
                    )
                record = _replayable(unknown[0])
            else:
                for name in names:
                    if (destination / name).exists() and not overwrite:
                        raise ValueError(
                            "Retrieval destination exists; explicitly request overwrite."
                        )
                for item in overlapping:
                    if not overwrite:
                        raise ValueError("A retrieval already targets these paths.")
                    if self.manager(target).check_transfer_status(item["transfer_id"])["status"] not in {"SUCCEEDED", "FAILED"}:
                        raise ValueError("A retrieval targeting these paths is still active; wait before overwriting.")
            with _transfer_phase("prepare"):
                manager = self.manager(target)
                prepared = manager.prepare_mapping(
                    mapping, reverse=True, label=manifest["identity"] + " results",
                    submission_id=record and record["submission_id"],
                )
            for name in names:
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
            replay = record is not None
            if not replay:
                record = {"files": names, "mapping": mapping}
                history.append(record)
            self._submit_transfer(
                manager, prepared, record, lambda: write_json(history_path, history),
                replay=replay,
            )
            return record

    def effective_settings(self, target_name, request):
        """Scheduler settings a request selects over the configured target defaults.

        Defaults come from the configuration, never from the editable run
        manifest, and only explicitly supplied resource fields override them.
        """
        target = self._target(target_name)
        resources = target.resources
        if request.resources is not None:
            resources = resources.model_copy(
                update=request.resources.model_dump(exclude_unset=True)
            )
        return {
            "project": request.project or target.project,
            "queue": request.queue or target.queue,
            "resources": resources.model_dump(),
        }

    def _jobspec(self, manifest, request):
        remote = manifest["remote_directory"]
        settings = self.effective_settings(manifest["target"], request)
        resources = settings["resources"]
        return {
            "executable": "/bin/bash",
            "arguments": [remote + "/" + request.launch_script, *request.arguments],
            "directory": remote,
            "name": manifest["identity"][:15],
            "stdout_path": remote + "/" + request.stdout,
            "stderr_path": remote + "/" + request.stderr,
            "environment": {},
            "resources": {"node_count": resources["node_count"]},
            "attributes": {
                "duration": resources["duration"],
                "queue_name": settings["queue"],
                "account": settings["project"],
                "custom_attributes": {"filesystems": resources["filesystems"]},
            },
        }

    @staticmethod
    def _summary(evidence, **extra):
        return {**evidence["attempts"][-1], "attempt_count": len(evidence["attempts"]), **extra}

    def submit(self, run_dir, request, *, resubmit=False):
        """Send each attempt at most once; resubmit only after a confirmed rejection.

        Transitions, all under the run lock:
        1. The new attempt is prepared in memory, including credentials.
        2. It is appended as ``prepared`` and saved. A crash here leaves the
           marker naming an older attempt (or none): nothing was sent, and the
           next call sends this attempt.
        3. The marker is atomically replaced by this attempt's ID. From now on
           the attempt may have been sent, so it is never sent again; a crash
           leaves it ``prepared`` with a matching marker, which status
           reconciles as unknown.
        4. The request is sent once and its outcome saved.
        """
        request = BatchRequest.model_validate(request)
        with self._run(run_dir) as root:
            manifest, target = self._load(root)
            if request.target != manifest["target"]:
                raise ValueError("Request target differs from this run.")
            intent = {
                # Only explicit fields, so a reload keeps configured defaults distinct.
                "batch": request.model_dump(exclude_unset=True),
                "jobspec": self._jobspec(manifest, request),
            }
            path = root / "submission.json"
            evidence = read_json(path) if path.exists() else {
                "manifest_hash": fingerprint(manifest), "attempts": [],
            }
            current = evidence["attempts"][-1] if evidence["attempts"] else None
            sent = current is not None and started_attempt(root) == current["id"]
            if current is not None and (sent or current["state"] != "prepared"):
                # Check input identity even for an accepted repeated request.
                self._check_inputs(root, manifest)
                if current["state"] in ("prepared", "submission_unknown"):
                    return self._summary(
                        evidence, state="submission_unknown", retry_safe=False,
                        message="Acceptance is unknown. Reconcile with hpc_job_status; never resubmit.",
                    )
                if current["state"] != "rejected":
                    # An accepted attempt keeps its recorded specification:
                    # match the same request or the same effective job.
                    if all(current["intent"][key] != intent[key] for key in ("batch", "jobspec")):
                        raise ValueError("Run specification changed; use a fresh run directory.")
                    return self._summary(evidence)
                if not resubmit:
                    return self._summary(
                        evidence, retry_safe=True,
                        message="The scheduler rejected this attempt. Correct the cause, then call "
                        "hpc_submit_job with resubmit=true in this run directory.",
                    )
                # A saved rejection can be stale: require scheduler history to
                # show no job for this run before sending another attempt.
                matches, complete = self._search(manifest, target, evidence)
                if matches or not complete:
                    raise ValueError(
                        "Scheduler history does not rule out an existing job for this run; "
                        "inspect it with hpc_job_status and hpc_list_jobs before resubmitting."
                    )
                current = None
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
            prepared = self.iri.prepare_submission(target.compute_resource, intent["jobspec"])
            if current is None:
                current = {"id": uuid.uuid4().hex, "created_at": now()}
                evidence["attempts"].append(current)
            # An unsent prepared attempt takes the request reviewed for this call.
            current.update(state="prepared", intent=intent)
            write_json(path, evidence)
            mark_started(root, current["id"])
            try:
                response = self.iri.submit_prepared(prepared)
                if isinstance(response, dict) and response.get("id"):
                    current.update(state="accepted", job_id=str(response["id"]))
                else:
                    current.update(state="submission_unknown")
                    if isinstance(response, dict) and response.get("task_id"):
                        current["operation_id"] = str(response["task_id"])
            except Exception as exc:
                definitive = (
                    isinstance(exc, IRIRequestError)
                    and exc.status_code in _DEFINITIVE_REJECTIONS
                )
                current.update(
                    state="rejected" if definitive else "submission_unknown",
                    error={"type": type(exc).__name__},
                )
                if isinstance(exc, IRIRequestError):
                    current["error"]["http_status"] = exc.status_code
                    current["error"]["authentication_required"] = exc.status_code == 401
            current["updated_at"] = now()
            write_json(path, evidence)
            if current["state"] == "rejected":
                return self._summary(
                    evidence, retry_safe=True,
                    message=(
                        f"{IRIRequestError(current['error']['http_status'])} The job was not accepted; "
                        "correct the cause, then call hpc_submit_job with resubmit=true."
                    ),
                )
            return self._summary(evidence)

    @staticmethod
    def _check_inputs(root, manifest):
        for name, digest in manifest["files"].items():
            if (
                checksum(_input_path(root, name)) != digest
                or checksum(root / ".hpc-inputs" / name) != digest
            ):
                raise ValueError("Staged inputs changed; use a fresh run directory.")

    def _search(self, manifest, target, evidence):
        """Find scheduler jobs for this run under every attempt's account.

        Returns the matches and whether the bounded search was complete.
        Every attempt uses the same job name and directory, so jobs from
        different attempts are indistinguishable and remain ambiguous.
        """
        expected = evidence["attempts"][-1]["intent"]["jobspec"]
        accounts = sorted({
            attempt["intent"]["jobspec"]["attributes"]["account"]
            for attempt in evidence["attempts"]
        })
        matches = {}
        for account in accounts:
            for historical in (False, True):
                for offset in range(0, 1000, 100):
                    records = self.iri.jobs(
                        target.compute_resource,
                        historical=historical,
                        include_spec=True,
                        limit=100,
                        offset=offset,
                        filters={"accountingId": account},
                    )
                    if not isinstance(records, list):
                        return matches, False  # Unknown shapes never establish identity.
                    for record in records:
                        spec = record.get("job_spec") if isinstance(record, dict) else None
                        if (
                            isinstance(spec, dict)
                            and record.get("id")
                            and spec.get("name") == expected["name"]
                            and spec.get("directory") == manifest["remote_directory"]
                            and spec.get("attributes", {}).get("account") == account
                        ):
                            matches[str(record["id"])] = record
                    if len(records) < 100:
                        break
                else:
                    return matches, False  # Search bound exceeded: matching is incomplete.
        return matches, True

    def _reconcile(self, manifest, target, evidence):
        current = evidence["attempts"][-1]
        if current.get("operation_id"):
            try:
                operation = self.iri.task(current["operation_id"])
            except IRIRequestError as exc:
                if exc.status_code != 404:
                    raise
                # A missing task handle says nothing about the job: rely on
                # scheduler history alone.
                operation = None
                current["operation_diagnostic"] = {"at": now(), "http_status": 404}
            if isinstance(operation, dict) and operation.get("status") == "failed":
                # The task may have failed after the scheduler accepted the job,
                # so failure is recorded but never treated as a rejection.
                current["operation_diagnostic"] = {"at": now(), "status": "failed"}
            result = operation.get("result") if isinstance(operation, dict) else None
            if (
                isinstance(result, dict)
                and operation.get("status") == "completed"
                and result.get("id")
            ):
                current.update(
                    state="accepted", job_id=str(result["id"]), reconciled_at=now()
                )
                return
            if isinstance(operation, dict) and operation.get("status") in (
                "pending",
                "active",
            ):
                current["state"] = "submission_unknown"
                return
        matches, complete = self._search(manifest, target, evidence)
        # An incomplete search could hide a second match, so it is ambiguous.
        if complete and len(matches) == 1:
            current.update(
                state="accepted", job_id=next(iter(matches)), reconciled_at=now()
            )
            return
        current["state"] = "submission_unknown"
        # No match is a diagnostic only: visibility and retention are not
        # guaranteed, so it never proves that the request was not accepted.
        current["last_search"] = {
            "at": now(), "complete": complete, "matches": sorted(matches),
        }

    def status(self, run_dir):
        with self._run(run_dir) as root:
            manifest, target = self._load(root)
            path = root / "submission.json"
            if not path.exists():
                return {
                    "state": "submission_unknown"
                    if (root / "submission.started").exists()
                    else "not_submitted"
                }
            evidence = read_json(path)
            current = evidence["attempts"][-1]
            if current["state"] in ("prepared", "submission_unknown"):
                if started_attempt(root) != current["id"]:
                    return self._summary(
                        evidence, message="This attempt was never sent; call hpc_submit_job to send it.",
                    )
                self._reconcile(manifest, target, evidence)
                write_json(path, evidence)
            if current["state"] != "accepted":
                return self._summary(
                    evidence, retry_safe=current["state"] == "rejected",
                )
            job_id = current["job_id"]
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
            if len(evidence["attempts"]) > 1:
                # Earlier attempts were rejected, but report any job they left.
                matches, complete = self._search(manifest, target, evidence)
                status["duplicate_jobs"] = sorted(set(matches) - {job_id})
                status["duplicate_search_complete"] = complete
            write_json(root / "job-status.json", status)
            return status

    def list_jobs(self, target, **kwargs):
        resource = self.iri.resolve_resource(self._target(target).compute_resource)
        return self.iri.jobs(resource, **kwargs)

    def cancel(self, run_dir, job_id):
        with self._run(run_dir) as root:
            manifest, target = self._load(root)
            path = root / "submission.json"
            if not path.exists():
                raise ValueError("No submission is recorded for this run.")
            evidence = read_json(path)
            current = evidence["attempts"][-1]
            accepted = current["state"] == "accepted" and current.get("job_id") == job_id
            # Any other job, such as a duplicate left by an earlier attempt, is
            # cancelled only when scheduler history shows this exact ID with
            # this run's name, directory and an attempt's account.
            if not accepted and job_id not in self._search(manifest, target, evidence)[0]:
                raise ValueError(
                    "Cancellation must match this run's full accepted job ID or a job "
                    "that scheduler history verifies belongs to this run."
                )
            # A cancellation request is not proof that the job has stopped.
            cancellation = {
                "state": "requested", "at": now(),
                "verified_by": "accepted_attempt" if accepted else "scheduler_history",
            }
            evidence.setdefault("cancellations", {})[job_id] = cancellation
            write_json(path, evidence)
            self.iri.cancel(target.compute_resource, job_id)
            cancellation["state"] = "acknowledged"
            write_json(path, evidence)
            return {"job_id": job_id, **cancellation}

    @staticmethod
    def _inspection(remote, operation, result, **extra):
        truncated = isinstance(result, dict) and bool(result.get("truncated"))
        response = {
            "path": remote, "operation": operation, "response": result,
            "truncated": truncated, **extra,
        }
        if isinstance(result, dict) and result.get("task_id"):
            response["message"] = (
                "The facility returned a task; poll it with hpc_operation_status using this run_dir."
            )
        elif truncated:
            response["message"] = "The response was truncated; retrieve the file with Globus for full content."
        return response

    def inspect(self, run_dir, path=".", *, operation="view", offset=0, size=16384):
        with self._run(run_dir) as root:
            manifest, target = self._load(root)
            if path != ".":
                path = relative_path(path)
            remote = str(PurePosixPath(manifest["remote_directory"]) / path)
            result = self.iri.inspect(
                target.storage_resource, remote, operation=operation, offset=offset, size=size,
            )
            if isinstance(result, dict) and result.get("task_id"):
                # Record the approved request so polling reproduces its bounds.
                operations_path = root / ".hpc-operations.json"
                operations = read_json(operations_path) if operations_path.exists() else {}
                operations[str(result["task_id"])] = {
                    "operation": operation, "path": remote, "offset": offset,
                    "size": size, "created_at": now(),
                }
                write_json(operations_path, operations)
            return self._inspection(remote, operation, result)

    def operation_status(self, run_dir, operation_id):
        """Poll a facility task that an approved inspection of this run returned."""
        with self._run(run_dir) as root:
            manifest, target = self._load(root)
            operations_path = root / ".hpc-operations.json"
            operations = read_json(operations_path) if operations_path.exists() else {}
            saved = operations.get(operation_id) if isinstance(operations, dict) else None
            if not isinstance(saved, dict) or not (
                saved.get("path") == manifest["remote_directory"]
                or str(saved.get("path", "")).startswith(manifest["remote_directory"] + "/")
            ):
                raise ValueError("Only operations returned by this run's inspections can be polled.")
            result = self.iri.inspect(
                target.storage_resource, saved["path"], operation=saved["operation"],
                operation_id=operation_id, offset=saved["offset"], size=saved["size"],
            )
            return self._inspection(
                saved["path"], saved["operation"], result, operation_id=operation_id,
            )
