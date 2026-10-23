"""Submit RUN_DIR/request.json to the ALCF IRI v2 API at most once.

request.json holds {"resource_id": "<compute resource UUID>", "job_spec": {...}}.
Evidence written to RUN_DIR:

  submission.started         exclusive attempt marker: idempotency key, request hash
  job.json, job.id           accepted job response and its full scheduler ID
  submission.uncertain.json  outcome unknown; inspect the scheduler before any retry
  submission.rejected.json   initial attempt rejected; no job was created

Exit codes: 0 submitted or already submitted, 1 rejected or invalid input,
3 uncertain. Only --retry-uncertain resends, with the original key and request.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid

BASE_URL = "https://api.alcf.anl.gov/api/v2"
TIMEOUT_S = 60.0
REJECTED = {400, 401, 403, 404, 405, 422}
UNCERTAIN = 3


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # Following a redirect would resend the token and key; report the 3xx instead.
    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def get_token():
    from alcf_tokens.auth import get_access_token

    return get_access_token("iri")


def _write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _post(url, body, key, token):
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Idempotency-Key": key,
    })
    try:
        with _OPENER.open(request, timeout=TIMEOUT_S) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _uncertain(run, record):
    _write(run / "submission.uncertain.json", json.dumps(record, indent=2))
    print(
        "Submission outcome is uncertain. List your jobs before any retry; "
        "never delete submission.started.", file=sys.stderr,
    )
    return UNCERTAIN


def submit(run_dir, *, retry_uncertain=False, base_url=BASE_URL):
    run = Path(run_dir)
    job_id = run / "job.id"
    marker = run / "submission.started"
    if job_id.exists():
        print(job_id.read_text().strip())
        print("Already submitted; monitor this job instead.", file=sys.stderr)
        return 0

    raw = (run / "request.json").read_bytes()
    request = json.loads(raw)
    resource = request.get("resource_id") if isinstance(request, dict) else None
    if not isinstance(resource, str) or not resource or not isinstance(
        request.get("job_spec"), dict
    ):
        print("request.json needs resource_id and a job_spec object.", file=sys.stderr)
        return 1
    digest = hashlib.sha256(raw).hexdigest()

    if retry_uncertain:
        if not marker.exists():
            print("No uncertain attempt to retry.", file=sys.stderr)
            return 1
        attempt = json.loads(marker.read_text())
        if attempt.get("request_sha256") != digest:
            print("request.json changed since the uncertain attempt; refusing.",
                  file=sys.stderr)
            return 1
    elif marker.exists():
        print("Submission already attempted; inspect before retrying.", file=sys.stderr)
        return UNCERTAIN
    try:
        token = get_token()
    except Exception as error:
        token = None
        print(f"Token retrieval failed: {error}", file=sys.stderr)
    if not token:
        print("IRI token unavailable; run: alcf-tokens login iri", file=sys.stderr)
        return 1
    if not retry_uncertain:
        attempt = {
            "idempotency_key": str(uuid.uuid4()),
            "request_sha256": digest,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if job_id.exists():
                print(job_id.read_text().strip())
                return 0
            print("Submission already attempted; inspect before retrying.",
                  file=sys.stderr)
            return UNCERTAIN
        with os.fdopen(descriptor, "w") as handle:
            json.dump(attempt, handle, indent=2)

    url = f"{base_url.rstrip('/')}/compute/job/{resource}"
    body = json.dumps(request["job_spec"]).encode()
    try:
        status, content = _post(url, body, attempt["idempotency_key"], token)
    except Exception as error:  # Any transport failure may follow acceptance.
        return _uncertain(run, {"error": repr(error)})
    try:
        response = json.loads(content)
    except ValueError:
        response = content.decode(errors="replace")
    job = response.get("id") if isinstance(response, dict) else None
    if 200 <= status < 300 and isinstance(job, str) and job.strip():
        _write(run / "job.json", json.dumps(response, indent=2))
        _write(job_id, job + "\n")
        print(job)
        return 0
    record = {"status": status, "response": response}
    if status in REJECTED and not retry_uncertain:
        _write(run / "submission.rejected.json", json.dumps(record, indent=2))
        marker.unlink()
        print(f"Rejected with HTTP {status}; fix the cause and run again.",
              file=sys.stderr)
        return 1
    return _uncertain(run, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--retry-uncertain", action="store_true",
                        help="Resend the uncertain attempt with its original key.")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    return submit(args.run_dir, retry_uncertain=args.retry_uncertain,
                  base_url=args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
