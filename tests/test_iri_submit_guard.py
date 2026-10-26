"""Exercise the iri-hpc scripts: the submission guard against a local fake IRI
server, target configuration, and the target loader in iri.sh."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tomllib

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/chemgraph/skills/iri-hpc/scripts/iri_submit.py"
)
REQUEST = {"resource_id": "polaris-id", "job_spec": {"executable": "/bin/bash"}}
PAGES = {
    "/status/resources": [
        {"name": "Polaris", "resource_type": "urn:doe-iri:resource:compute", "id": "polaris-id"},
        {"name": "Eagle", "resource_type": "urn:doe-iri:resource:storage", "id": "eagle-id"},
    ],
    "/account/projects": [
        {"name": "A", "user_ids": ["me", "x"]},
        {"name": "B", "user_ids": ["me", "y"]},
    ],
}


@pytest.fixture
def guard(monkeypatch):
    spec = importlib.util.spec_from_file_location("iri_submit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_token", lambda: "secret-token")
    monkeypatch.setattr(module, "TIMEOUT_S", 0.5)
    return module


@pytest.fixture
def server():
    """Reply with queued (status, body) pairs; "slow" outlasts the client timeout.

    A 302 points at /elsewhere; a followed redirect would show up as a GET.
    """
    replies, posts = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            posts.append({"path": self.path, "headers": dict(self.headers), "body": body})
            status, reply = replies.pop(0)
            if reply == "slow":
                time.sleep(1.5)
                status, reply = 200, {"id": "late.polaris-pbs-01"}
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "/elsewhere")
            self.end_headers()
            try:
                self.wfile.write(json.dumps(reply).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            posts.append({"path": self.path, "headers": dict(self.headers), "body": b""})
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/api/v2", replies, posts
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def run(tmp_path):
    (tmp_path / "request.json").write_text(json.dumps(REQUEST))
    return tmp_path


def test_success_records_job_and_never_resubmits(guard, server, run, capsys):
    url, replies, posts = server
    job = "123.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    replies.append((200, {"id": job}))
    assert guard.submit(run, base_url=url) == 0
    assert capsys.readouterr().out == job + "\n"
    assert (run / "job.id").read_text().strip() == job
    assert json.loads((run / "job.json").read_text())["id"] == job
    assert posts[0]["path"] == "/api/v2/compute/job/polaris-id"
    assert json.loads(posts[0]["body"]) == REQUEST["job_spec"]
    assert not any(k.lower() == "idempotency-key" for k in posts[0]["headers"])
    marker = json.loads((run / "submission.started").read_text())
    assert marker["idempotency_header_sent"] is False and "idempotency_key" not in marker
    assert guard.submit(run, base_url=url) == 0 and len(posts) == 1


@pytest.mark.parametrize("reply", [
    (500, {"title": "error"}), (409, {}), (200, "slow"), (200, {}), (302, {}),
])
def test_uncertainty_and_deprecated_retry_never_resend(guard, server, run, reply):
    url, replies, posts = server
    replies.append(reply)
    assert guard.submit(run, base_url=url) == 3
    marker = (run / "submission.started").read_bytes()
    assert not (run / "job.id").exists()
    assert (run / "submission.uncertain.json").exists()
    assert guard.submit(run, base_url=url) == 3
    assert guard.submit(run, retry_uncertain=True, base_url=url) == 3
    (run / "request.json").write_text(json.dumps({**REQUEST, "resource_id": "aurora-id"}))
    assert guard.submit(run, retry_uncertain=True, base_url=url) == 3
    assert (run / "submission.started").read_bytes() == marker and len(posts) == 1


def test_initial_rejection_allows_fixed_resubmission(guard, server, run, capsys):
    url, replies, posts = server
    replies += [(422, {"detail": "invalid queue"}), (200, {"id": "9.polaris"})]
    assert guard.submit(run, base_url=url) == 1
    assert "422" in capsys.readouterr().err
    assert not (run / "submission.started").exists()
    archive = next((run / "submission.history").glob("*.json"))
    original = archive.read_bytes()
    assert json.loads(original)["response"]["detail"] == "invalid queue"
    fixed = {**REQUEST, "job_spec": {**REQUEST["job_spec"], "attributes": {"queue_name": "debug"}}}
    (run / "request.json").write_text(json.dumps(fixed))
    assert guard.submit(run, base_url=url) == 0
    assert len(posts) == 2 and json.loads(posts[1]["body"]) == fixed["job_spec"]
    assert archive.read_bytes() == original


def test_concurrent_attempts_send_one_request(guard, server, run):
    url, replies, posts = server
    replies.append((200, {"id": "5.polaris"}))
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(guard.submit(run, base_url=url)))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(posts) == 1 and 0 in codes and set(codes) <= {0, 3}


@pytest.mark.parametrize("damage", ["token", "request", "resource"])
def test_preflight_does_not_claim_an_attempt(guard, run, monkeypatch, damage):
    if damage == "token":
        monkeypatch.setattr(guard, "get_token", lambda: None)
    elif damage == "request":
        (run / "request.json").write_text("[]")
    else:
        (run / "request.json").write_text(json.dumps({**REQUEST, "resource_id": ".."}))
    monkeypatch.setattr(guard, "_post", lambda *a: pytest.fail("preflight must not POST"))
    assert guard.submit(run) == 1
    assert not (run / "submission.started").exists()


def fail():
    raise OSError("injected persistence failure")


@pytest.mark.parametrize("name,content", [
    ("job.id", ""), ("submission.started", "{}"),
    ("submission.uncertain.json", "{}"), ("submission.history", "broken"),
])
def test_corrupt_or_orphaned_state_blocks(guard, run, monkeypatch, name, content):
    path = run / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(content)
    monkeypatch.setattr(guard, "_post", lambda *a: pytest.fail("corrupt state must not POST"))
    assert guard.submit(run) == 3


def test_marker_sync_failure_sends_nothing_and_allows_later_submit(guard, run, server, monkeypatch):
    url, replies, posts = server
    sync_dir = guard._sync_dir
    calls = 0

    def fail_after_publish(path):
        nonlocal calls
        calls += 1
        if calls == 2:  # Directory sync after the complete marker is published.
            fail()
        return sync_dir(path)

    with monkeypatch.context() as patch:
        patch.setattr(guard, "_sync_dir", fail_after_publish)
        assert guard.submit(run, base_url=url) == 3
    assert not posts and not (run / "submission.started").exists()
    replies.append((200, {"id": "once"}))
    assert guard.submit(run, base_url=url) == 0 and len(posts) == 1


@pytest.mark.parametrize("target", ["history", "job.id", "submission.rejected.json"])
def test_post_persistence_failure_never_resends(guard, run, server, monkeypatch, capsys, target):
    url, replies, posts = server
    rejected = target == "submission.rejected.json"
    replies.append((422, {"detail": "bad queue"}) if rejected else (200, {"id": "once"}))
    original = guard._write

    def write(path, text, **kwargs):
        if path.name == target or (target == "history" and path.parent.name == "submission.history"):
            fail()
        return original(path, text, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(guard, "_write", write)
        assert guard.submit(run, base_url=url) == 3
    output = capsys.readouterr()
    assert output.out == ""
    assert ("bad queue" if rejected else "once") in output.err
    assert ("422" if rejected else "200") in output.err
    assert (run / "submission.started").exists() and len(posts) == 1
    expected = 3 if target == "history" else 1 if rejected else 0
    assert guard.submit(run, base_url=url) == expected and len(posts) == 1
    if rejected:
        assert not (run / "submission.started").exists()


@pytest.fixture
def legacy(guard, run):
    marker = json.dumps({"idempotency_key": "old-key", "request_sha256": guard._sha(
        (run / "request.json").read_text()), "started_at": "2026-10-23T00:00:00Z"}, indent=2)
    (run / "submission.started").write_bytes(marker.replace("\n", "\r\n").encode())
    outcome = {"status": 501, "response": {"status": 501, "detail": guard.CAPABILITY_ERROR}}
    (run / "submission.uncertain.json").write_text(json.dumps(outcome))
    note = run / "evidence.txt"
    note.write_text("Reviewed session A attempts 1 and 2: both rejected before scheduler submission.")
    return note


def test_legacy_resolution_is_local_repeatable_and_preserves_evidence(guard, run, legacy, monkeypatch, capsys, server):
    # Older runs may retain a prior rejection and extra fields in the latest error.
    (run / "submission.rejected.json").write_text(json.dumps({"status": 422}))
    error_path = run / "submission.uncertain.json"
    error_path.write_text(json.dumps({**json.loads(error_path.read_text()),
                                     "marker": "unrelated", "kind": "accepted"}))
    original = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    with monkeypatch.context() as patch:
        patch.setattr(guard, "get_token", lambda: pytest.fail("recovery must not authenticate"))
        patch.setattr(guard, "_post", lambda *a: pytest.fail("recovery must not POST"))
        for _ in range(2):
            assert guard.resolve_rejection(run, legacy, confirmed=True) == 0
            output = capsys.readouterr()
            assert output.out == "" and "No job submitted" in output.err
    record = json.loads(next((run / "submission.history").glob("*.json")).read_text())
    assert record["kind"] == "resolved"
    assert record["marker"].encode() == original["submission.started"]
    assert record["snapshots"]["submission.uncertain.json"].encode() == original["submission.uncertain.json"]
    assert record["evidence"] == legacy.read_text()
    for name, content in original.items():
        if name != "submission.started":
            assert (run / name).read_bytes() == content
    url, replies, posts = server
    replies.append((200, {"id": "recovered"}))
    assert guard.submit(run, base_url=url) == 0
    assert guard.submit(run, base_url=url) == 0 and len(posts) == 1
    assert guard.resolve_rejection(run, legacy, confirmed=True) == 3


@pytest.mark.parametrize("damage", ["changed-request", "generic-501", "accepted", "prior-timeout"])
def test_legacy_recovery_refuses_unsafe_evidence(guard, run, legacy, monkeypatch, damage):
    marker = run / "submission.started"
    if damage == "changed-request":
        (run / "request.json").write_text("{}")
    elif damage == "accepted":
        (run / "job.id").write_text("existing.polaris\n")
    elif damage == "prior-timeout":
        prior = marker.read_text().replace("old-key", "previous-key")
        guard._archive(run, prior, "uncertain", {"error": "timeout"}, {})
    else:
        (run / "submission.uncertain.json").write_text(json.dumps({
            "status": 501, "response": {"status": 501, "detail": "not implemented"},
        }))
    before = marker.read_bytes()
    monkeypatch.setattr(guard, "get_token", lambda: pytest.fail("recovery must not authenticate"))
    assert guard.resolve_rejection(run, legacy, confirmed=True) == 3
    assert marker.read_bytes() == before


def test_recovery_requires_evidence_and_attestation(guard, run, legacy):
    assert guard.resolve_rejection(run / "absent", legacy, confirmed=True) == 1
    assert guard.resolve_rejection(run, legacy) == 1
    assert guard.resolve_rejection(run, run / "absent", confirmed=True) == 1
    legacy.write_text(" ")
    assert guard.resolve_rejection(run, legacy, confirmed=True) == 1
    assert (run / "submission.started").exists()
    assert guard.main([str(run), "--resolve-idempotency-rejection", str(legacy),
                       "--confirm-all-attempts-rejected", "--base-url", "http://unused.invalid"]) == 1
    assert guard.main([str(run), "--confirm-all-attempts-rejected"]) == 1
    assert guard.main([str(run), "--retry-uncertain", "--resolve-idempotency-rejection", str(legacy)]) == 1


def test_empty_recovery_filename_never_submits(guard, run, monkeypatch, capsys):
    monkeypatch.setattr(guard, "get_token", lambda: pytest.fail("recovery must not authenticate"))
    assert guard.main([str(run), "--resolve-idempotency-rejection", ""]) == 1
    assert capsys.readouterr().out == ""
    assert not (run / "submission.started").exists() and not (run / "job.id").exists()


@pytest.mark.parametrize("boundary", ["archive", "retire"])
def test_recovery_interruption_and_new_marker(guard, run, legacy, monkeypatch, boundary):
    old = (run / "submission.started").read_bytes()
    with monkeypatch.context() as patch:
        if boundary == "archive":
            patch.setattr(guard, "_archive", lambda *a, **kw: fail())
        else:
            patch.setattr(guard, "_retire", lambda *a: fail())
        assert guard.resolve_rejection(run, legacy, confirmed=True) == 3
    assert guard.resolve_rejection(run, legacy, confirmed=True) == 0
    assert guard.resolve_rejection(run, legacy, confirmed=True) == 0
    newer = old.replace(b"old-key", b"newer-key")
    (run / "submission.started").write_bytes(newer)
    # A new marker with stale legacy error cannot inherit an earlier resolution.
    assert guard.resolve_rejection(run, legacy, confirmed=True) == 3
    assert (run / "submission.started").read_bytes() == newer


def test_headerless_capability_501_is_never_recoverable(guard, run, server):
    url, replies, posts = server
    replies.append((501, {"status": 501, "detail": guard.CAPABILITY_ERROR}))
    assert guard.submit(run, base_url=url) == 3
    note = run / "evidence.txt"
    note.write_text("Operator reviewed this rejection.")
    assert guard.resolve_rejection(run, note, confirmed=True) == 3
    assert guard.submit(run, retry_uncertain=True, base_url=url) == 3 and len(posts) == 1


def test_recovery_and_submission_share_lock(guard, run, legacy):
    with guard._lock(run):
        assert guard.resolve_rejection(run, legacy, confirmed=True) == 3
        assert guard.submit(run) == 3
    assert guard.resolve_rejection(run, legacy, confirmed=True) == 0


def test_real_submission_cli(run, server, monkeypatch):
    """Exercise submission through the agent's filtered environment on every CI OS."""
    import shlex
    from chemgraph.agent.deepagent_backend import create_host_shell_backend

    url, replies, posts = server
    replies.append((200, {"id": "cli.polaris"}))
    package = run / "alcf_tokens"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "auth.py").write_text('def get_access_token(service):\n    return "fake-token"\n')
    monkeypatch.setenv("PYTHONPATH", str(run) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    argv = [sys.executable, str(SCRIPT), str(run), "--base-url", url]
    command = subprocess.list2cmdline(argv) if sys.platform == "win32" else shlex.join(argv)
    backend = create_host_shell_backend(run)
    for _ in range(2):
        result = backend.execute(command, timeout=30)
        assert result.exit_code == 0, result.output
        assert "cli.polaris" in result.output
    assert len(posts) == 1 and (run / "job.id").read_text() == "cli.polaris\n"


@pytest.fixture
def configure(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("iri_configure", SCRIPT.with_name("iri_configure.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_token", lambda: "secret-token")
    monkeypatch.setattr(module, "_get", lambda url, token=None: PAGES[url.split("/api/v2")[1]])
    monkeypatch.setattr(module, "local_collection", lambda: "laptop-id")
    return module


def test_configure_asks_once_then_writes_an_eagle_target(configure, tmp_path, capsys):
    assert configure.eagle_root("/lus/eagle/projects/P/u") == "/eagle/P/u"
    assert configure.eagle_root("/lus/eagle/other/u") is None
    assert configure.eagle_root("/eagle/../home/u") is None
    tmp_path = tmp_path / "w-\N{GRINNING FACE}"
    tmp_path.mkdir()
    assert configure.configure(tmp_path) == 2
    assert json.loads(capsys.readouterr().out) == {
        "ask": ["project", "environment_setup", "python"],
        "choices": {"project": ["A", "B"]},
    }
    assert configure.configure(tmp_path, remote_root="/home/me/runs") == 1
    assert not (tmp_path / "target.toml").exists()

    answers = {"project": "A", "environment_setup": "source env/bin/activate",
               "python": "env/bin/python"}
    assert configure.configure(tmp_path, **answers) == 0
    target = tomllib.loads((tmp_path / "target.toml").read_text(encoding="utf-8"))
    assert target["username"] == "me" and target["storage_resource_id"] == "eagle-id"
    assert target["remote_root"] == "/eagle/A/me/chemgraph-runs"
    assert target["remote_globus_root"] == "/A/me/chemgraph-runs"
    assert target["local_globus_root"].endswith("w-\N{GRINNING FACE}")
    assert configure.configure(tmp_path, **answers) == 1


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="needs bash")
def test_iri_sh_loads_target_and_keeps_overrides(tmp_path):
    (tmp_path / "target.toml").write_text(
        'compute_resource_id = "c"\nstorage_resource_id = "s"\nusername = "u $(touch pwned)"\n'
    )
    loader = SCRIPT.with_name("iri.sh").as_posix()
    script = (f'set -euo pipefail; STORAGE=override; source "{loader}"; '
              'printf "%s|%s|%s" "$COMPUTE" "$STORAGE" "$OWNER"')
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    out = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True).stdout
    assert out == "c|override|u $(touch pwned)"
    assert not (tmp_path / "pwned").exists()

    (tmp_path / "target.toml").write_text("compute_resource_id = \n")
    broken = subprocess.run(["bash", "-c", f'source "{loader}" || echo stopped'], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert broken.stdout.strip() == "stopped" and "Could not read target.toml" in broken.stderr
