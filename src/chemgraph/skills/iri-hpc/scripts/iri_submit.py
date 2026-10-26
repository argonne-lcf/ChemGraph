"""Submit RUN_DIR/request.json once through the ordinary ALCF IRI v2 API.

No Idempotency-Key is sent. submission.started blocks an uncertain submission;
job.id/job.json identify acceptance. Immutable submission.history records preserve
outcomes before markers are retired. The latest uncertain/rejected JSON files
remain convenient snapshots, not permission to retry.

Exit codes: 0 accepted/already accepted, 1 invalid input/definite rejection,
3 uncertain/busy/inconsistent state. --retry-uncertain never resends.
--resolve-idempotency-rejection is local recovery only: exit 0, empty stdout,
and no job submission. It requires evidence and explicit operator confirmation.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE_URL = "https://api.alcf.anl.gov/api/v2"
TIMEOUT_S = 60.0
REJECTED = {400, 401, 403, 404, 405, 422}
UNCERTAIN = 3
CAPABILITY_ERROR = (
    "Idempotency-Key provided but no idempotency store is configured "
    "(set IRI_IDEMPOTENCY_STORE)."
)
SNAPSHOTS = ("submission.uncertain.json", "submission.rejected.json")


class StateError(ValueError):
    """Saved evidence does not permit another submission."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


# iri_configure.py also imports these names.
_OPENER = urllib.request.build_opener(_NoRedirect)


def get_token():
    from alcf_tokens.auth import get_access_token

    return get_access_token("iri")


def _say(message, *, stdout=False):
    stream = sys.stdout if stdout else sys.stderr
    try:
        print(message, file=stream, flush=True)
    except BrokenPipeError:
        # A timed-out agent may close its pipes while this process finishes.
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), stream.fileno())
        except (OSError, ValueError):
            pass


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path):
    return path.read_bytes().decode("utf-8")


def _object(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise StateError("Expected a JSON object in saved evidence.")
    return value


def _marker(text):
    value = _object(text)
    if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("request_sha256", ""))):
        raise StateError("Invalid submission marker request hash.")
    legacy_key = value.get("idempotency_key")
    attempt_id = value.get("attempt_id")
    legacy = (isinstance(legacy_key, str) and legacy_key.strip() and
              value.get("idempotency_header_sent", True) is True)
    current = (legacy_key is None and isinstance(attempt_id, str) and attempt_id.strip() and
               value.get("idempotency_header_sent") is False)
    if not isinstance(value.get("started_at"), str) or not value["started_at"] or not (legacy or current):
        raise StateError("Invalid submission marker; inspect saved evidence.")
    return value


def _job_id(value):
    return isinstance(value, str) and re.fullmatch(r"\S+", value) is not None


def _sync_dir(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write(path, text, *, exclusive=False, rollback=False):
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        _sync_dir(path.parent)  # Reject unsupported directory sync before publishing.
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        published = True
        _sync_dir(path.parent)
    except OSError:
        if rollback and published and _read(path) == text:
            path.unlink()  # This invocation has not handed any request to urllib.
            try:
                _sync_dir(path.parent)
            except OSError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(run):
    with (run / ".submission.lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError) as error:
            raise StateError("Submission/recovery is busy or locking is unavailable.") from error
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _capability(record):
    response = record.get("response")
    return (
        record.get("status") == 501 and isinstance(response, dict) and
        response.get("status") == 501 and response.get("detail") == CAPABILITY_ERROR
    )


def _history(run):
    records = {}
    directory = run / "submission.history"
    if directory.exists() and not directory.is_dir():
        raise StateError("Invalid submission history directory.")
    for path in sorted(directory.glob("*.json")):
        record = _object(_read(path))
        marker = record.get("marker", "")
        attempt = _marker(marker)
        if _sha(marker) != path.stem or record.get("kind") not in {
            "accepted", "rejected", "uncertain", "resolved"
        }:
            raise StateError("Invalid submission history.")
        snapshots = record.get("snapshots")
        if not isinstance(snapshots, dict) or any(
            name not in SNAPSHOTS or not isinstance(text, str) for name, text in snapshots.items()
        ):
            raise StateError("Invalid history snapshots.")
        for text in snapshots.values():
            _object(text)
        kind = record["kind"]
        if kind == "accepted" and not (
            200 <= record.get("status", 0) < 300 and
            isinstance(record.get("response"), dict) and
            _job_id(record["response"].get("id"))
        ):
            raise StateError("Invalid accepted history.")
        if kind == "rejected" and record.get("status") not in REJECTED:
            raise StateError("Invalid rejected history.")
        if kind in {"rejected", "resolved"}:
            name = "submission.rejected.json" if kind == "rejected" else "submission.uncertain.json"
            snapshot = _object(snapshots.get(name, "{}"))
            if any(snapshot.get(key) != record.get(key) for key in ("status", "response")):
                raise StateError("History and saved outcome disagree.")
        if kind == "resolved" and not (
            attempt.get("idempotency_key") and _capability(record) and
            isinstance(record.get("evidence"), str) and record["evidence"].strip() and
            _sha(record["evidence"]) == record.get("evidence_sha256") and
            isinstance(record.get("covered"), list)
        ):
            raise StateError("Invalid legacy resolution history.")
        records[path.stem] = record
    for record in records.values():
        for key in record.get("covered", []):
            prior = records.get(key, {})
            if not (prior.get("kind") == "uncertain" and _capability(prior) and
                    _marker(prior["marker"]).get("idempotency_key")):
                raise StateError("Resolution cannot clear a different uncertain attempt.")
    return records


def _snapshots(run):
    saved = {name: _read(run / name) for name in SNAPSHOTS if (run / name).exists()}
    for name, text in saved.items():
        value = _object(text)
        if name == "submission.rejected.json" and value.get("status") not in REJECTED:
            raise StateError("Invalid rejection snapshot.")
    return saved


def _clear_history(records, snapshots):
    covered = {key for r in records.values() if r["kind"] == "resolved" for key in r["covered"]}
    if any(r["kind"] == "accepted" or (r["kind"] == "uncertain" and k not in covered)
           for k, r in records.items()):
        raise StateError("Unresolved submission history; no request sent.")
    uncertain = snapshots.get("submission.uncertain.json")
    if uncertain is not None and not any(
        r["snapshots"].get("submission.uncertain.json") == uncertain for r in records.values()
    ):
        raise StateError("Uncertain snapshot has no matching history; no request sent.")


def _archive(run, marker, kind, outcome, snapshots, **extra):
    directory = run / "submission.history"
    directory.mkdir(exist_ok=True)
    _sync_dir(run)
    record = {**outcome, "marker": marker, "kind": kind, "snapshots": snapshots,
              "completed_at": datetime.now(timezone.utc).isoformat(), **extra}
    _write(directory / f"{_sha(marker)}.json", json.dumps(record, indent=2), exclusive=True)
    return record


def _retire(run, marker):
    path = run / "submission.started"
    if _read(path) != marker:
        raise StateError("Active marker changed; refusing to clear it.")
    path.unlink()
    _sync_dir(run)


def _accept(run, record):
    response = record["response"]
    job = response["id"]
    if (run / "job.json").exists() and _object(_read(run / "job.json")).get("id") != job:
        raise StateError("Conflicting accepted job response.")
    _write(run / "job.json", json.dumps(response, indent=2))
    _write(run / "job.id", job + "\n")
    _say(job, stdout=True)
    return 0


def _rejected(run, marker, record):
    _write(run / "submission.rejected.json", record["snapshots"]["submission.rejected.json"])
    _retire(run, marker)
    _say(f"Rejected with HTTP {record['status']}: {record.get('response')}. "
         "Fix the cause and submit separately.")
    return 1


def _post(request):
    try:
        with _OPENER.open(request, timeout=TIMEOUT_S) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _submit(run, base_url):
    if (run / "job.id").exists():
        job = _read(run / "job.id").strip()
        if not _job_id(job):
            raise StateError("Invalid job.id; inspect saved evidence.")
        if (run / "job.json").exists() and _object(_read(run / "job.json")).get("id") != job:
            raise StateError("Conflicting accepted job response.")
        _say(job, stdout=True)
        _say("Already submitted; monitor this job instead.")
        return 0
    records, snapshots = _history(run), _snapshots(run)
    if (run / "submission.started").exists():
        marker = _read(run / "submission.started")
        _marker(marker)
        previous = records.get(_sha(marker), {})
        if previous.get("kind") == "accepted":
            return _accept(run, previous)  # Finish interrupted acceptance, never resend.
        if previous.get("kind") == "rejected":
            _clear_history(records, snapshots)
            return _rejected(run, marker, previous)
        raise StateError("Submission already attempted; inspect saved evidence. No request sent.")
    if (run / "job.json").exists():
        raise StateError("Job response exists without its active marker/ID; inspect it.")
    _clear_history(records, snapshots)
    try:
        raw = _read(run / "request.json")
        value = _object(raw)
        resource = value.get("resource_id")
        base = urllib.parse.urlsplit(base_url)
        if (not isinstance(resource, str) or not re.fullmatch(r"[\w.:-]+", resource, re.ASCII)
            or resource in {".", ".."} or not isinstance(value.get("job_spec"), dict) or
            base.scheme not in {"http", "https"} or not base.hostname or
            base.username or base.password or base.query or base.fragment or
            any(c.isspace() or ord(c) < 32 for c in base_url)):
            raise ValueError("request.json needs a plain resource_id and job_spec; base URL must be HTTP(S).")
        base.port  # Validate malformed ports before publishing a marker.
        token = get_token()
        if not token:
            raise ValueError("IRI token unavailable; run: alcf-tokens login iri")
        url = f"{base_url.rstrip('/')}/compute/job/{resource}"
        request = urllib.request.Request(url, data=json.dumps(value["job_spec"]).encode(),
                                         method="POST", headers={
                                             "Authorization": f"Bearer {token}",
                                             "Content-Type": "application/json",
                                         })
    except Exception as error:
        _say(f"Submission preflight failed: {error}")
        return 1
    marker = json.dumps({"attempt_id": str(uuid.uuid4()), "request_sha256": _sha(raw),
                         "started_at": datetime.now(timezone.utc).isoformat(),
                         "url": url, "idempotency_header_sent": False}, indent=2)
    _write(run / "submission.started", marker, exclusive=True, rollback=True)
    try:
        status, content = _post(request)
        try:
            response = json.loads(content)
        except ValueError:
            response = content.decode(errors="replace")
        outcome = {"status": status, "response": response}
    except Exception as error:
        outcome = {"error": repr(error)}
    try:
        status, response = outcome.get("status", 0), outcome.get("response")
        if 200 <= status < 300 and isinstance(response, dict) and _job_id(response.get("id")):
            record = _archive(run, marker, "accepted", outcome, snapshots)
            return _accept(run, record)
        kind = "rejected" if status in REJECTED else "uncertain"
        snapshot = json.dumps(outcome, indent=2)
        name = f"submission.{kind}.json"
        # Preserve the previous snapshot as well as the new one before replacing it.
        record = _archive(run, marker, kind, outcome, {**snapshots, name: snapshot},
                          previous_snapshots=snapshots)
        if kind == "rejected":
            return _rejected(run, marker, record)
        _write(run / name, snapshot)
        _say(f"Submission outcome is uncertain: {outcome}. No resend is permitted; inspect saved evidence.")
        return UNCERTAIN
    except (OSError, ValueError, TypeError, KeyError) as error:
        # The server may have accepted the job even if local persistence failed.
        _say(f"Could not save submission outcome {outcome!r}: {error}. "
             "Keep the marker; inspect this response before any further action.")
        return UNCERTAIN


def submit(run_dir, *, retry_uncertain=False, base_url=BASE_URL):
    if retry_uncertain:
        _say("--retry-uncertain is disabled: without server deduplication a resend may duplicate a job.")
        return UNCERTAIN
    try:
        run = Path(run_dir)
        if not run.is_dir():
            _say("Run directory does not exist.")
            return 1
        with _lock(run):
            return _submit(run, base_url)
    except (OSError, ValueError, TypeError, KeyError) as error:
        _say(f"Submission blocked; inspect saved evidence: {error}")
        return UNCERTAIN


def resolve_rejection(run_dir, evidence_file, *, confirmed=False):
    run = Path(run_dir)
    if not run.is_dir():
        _say("Run directory does not exist.")
        return 1
    if not confirmed:
        _say("Recovery requires --confirm-all-attempts-rejected after reviewing every attempt.")
        return 1
    try:
        evidence = _read(Path(evidence_file))
        if not evidence.strip():
            raise ValueError("Evidence note must not be empty.")
    except (OSError, ValueError) as error:
        _say(f"Invalid recovery evidence: {error}")
        return 1
    try:
        run = Path(run_dir)
        with _lock(run):
            if (run / "job.id").exists() or (run / "job.json").exists():
                raise StateError("Job acceptance evidence exists; recovery refused.")
            digest = _sha(_read(run / "request.json"))
            records, snapshots = _history(run), _snapshots(run)
            matching = [r for r in records.values() if r["kind"] == "resolved" and
                        _marker(r["marker"])["request_sha256"] == digest and
                        r["evidence_sha256"] == _sha(evidence)]
            path = run / "submission.started"
            if not path.exists():
                _clear_history(records, snapshots)
                if not matching:
                    raise StateError("No matching legacy resolution or marker.")
            else:
                marker = _read(path)
                attempt = _marker(marker)
                if any(r["kind"] == "resolved" and r["marker"] != marker for r in records.values()):
                    raise StateError("A prior resolution cannot clear a newer marker.")
                if (not isinstance(attempt.get("idempotency_key"), str) or
                    not attempt["idempotency_key"].strip() or
                    attempt.get("idempotency_header_sent", True) is not True or
                    attempt["request_sha256"] != digest):
                    raise StateError("Recovery requires an unchanged legacy headered request.")
                outcome = _object(snapshots.get("submission.uncertain.json", "{}"))
                if not _capability(outcome):
                    raise StateError("Only the exact missing-idempotency-store 501 can be resolved.")
                covered = []
                for key, record in records.items():
                    if record["kind"] == "uncertain":
                        if not (_capability(record) and _marker(record["marker"]).get("idempotency_key")):
                            raise StateError("An earlier uncertain attempt prevents recovery.")
                        covered.append(key)
                    elif record["kind"] == "accepted":
                        raise StateError("Accepted history prevents recovery.")
                previous = records.get(_sha(marker))
                if previous is not None:
                    if previous not in matching or previous["snapshots"] != snapshots:
                        raise StateError("Conflicting resolution history.")
                else:
                    _archive(run, marker, "resolved", outcome, snapshots, evidence=evidence,
                             evidence_sha256=_sha(evidence), covered=covered)
                _retire(run, marker)
            _say("Rejection resolved. No job submitted; run normal submission separately. Never restage.")
            return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        _say(f"Recovery blocked; existing evidence retained: {error}")
        return UNCERTAIN


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--retry-uncertain", action="store_true", help="Deprecated; never resends.")
    parser.add_argument("--resolve-idempotency-rejection", metavar="EVIDENCE_FILE")
    parser.add_argument("--confirm-all-attempts-rejected", action="store_true")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    recovery = args.resolve_idempotency_rejection is not None
    if ((recovery and (args.retry_uncertain or args.base_url is not None)) or
        (args.confirm_all_attempts_rejected and not recovery)):
        _say("Recovery, retry, and normal submission are separate actions.")
        return 1
    if recovery:
        if not args.resolve_idempotency_rejection:
            _say("Recovery evidence filename must not be empty.")
            return 1
        return resolve_rejection(args.run_dir, args.resolve_idempotency_rejection,
                                 confirmed=args.confirm_all_attempts_rejected)
    return submit(args.run_dir, retry_uncertain=args.retry_uncertain,
                  base_url=BASE_URL if args.base_url is None else args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
