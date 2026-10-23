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
            self.wfile.write(json.dumps(reply).encode())

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


@pytest.fixture
def run(tmp_path):
    (tmp_path / "request.json").write_text(json.dumps(REQUEST))
    return tmp_path


def test_success_records_job_and_never_resubmits(guard, server, run):
    url, replies, posts = server
    replies.append((200, {"id": "123.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"}))
    assert guard.submit(run, base_url=url) == 0
    assert (run / "job.id").read_text().strip() == "123.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
    assert json.loads((run / "job.json").read_text())["id"].startswith("123.")
    assert posts[0]["path"] == "/api/v2/compute/job/polaris-id"
    assert json.loads(posts[0]["body"]) == REQUEST["job_spec"]
    key = json.loads((run / "submission.started").read_text())["idempotency_key"]
    assert posts[0]["headers"]["Idempotency-Key"] == key
    assert guard.submit(run, base_url=url) == 0 and len(posts) == 1


@pytest.mark.parametrize(
    "reply", [(500, {"title": "error"}), (200, "slow"), (200, {}), (302, {})]
)
def test_uncertain_outcome_keeps_marker_and_blocks_resubmission(guard, server, run, reply):
    url, replies, posts = server
    replies.append(reply)
    assert guard.submit(run, base_url=url) == guard.UNCERTAIN
    assert (run / "submission.started").exists() and not (run / "job.id").exists()
    assert (run / "submission.uncertain.json").exists()
    assert guard.submit(run, base_url=url) == guard.UNCERTAIN and len(posts) == 1


def test_retry_reuses_key_and_refuses_changed_request(guard, server, run):
    url, replies, posts = server
    replies += [(503, {}), (200, {"id": "7.polaris"})]
    assert guard.submit(run, base_url=url) == guard.UNCERTAIN
    assert guard.submit(run, retry_uncertain=True, base_url=url) == 0
    first, second = posts
    assert first["headers"]["Idempotency-Key"] == second["headers"]["Idempotency-Key"]
    assert first["body"] == second["body"]

    other = run / "other"
    other.mkdir()
    (other / "request.json").write_text(json.dumps(REQUEST))
    replies.append((504, {}))
    assert guard.submit(other, base_url=url) == guard.UNCERTAIN
    (other / "request.json").write_text(json.dumps({**REQUEST, "resource_id": "aurora-id"}))
    assert guard.submit(other, retry_uncertain=True, base_url=url) == 1
    assert len(posts) == 3


def test_rejected_retry_does_not_resolve_lost_acceptance(guard, server, run):
    url, replies, posts = server
    replies += [(200, "slow"), (401, {"title": "token expired"})]
    assert guard.submit(run, base_url=url) == guard.UNCERTAIN
    key = json.loads((run / "submission.started").read_text())["idempotency_key"]
    assert guard.submit(run, retry_uncertain=True, base_url=url) == guard.UNCERTAIN
    assert json.loads((run / "submission.started").read_text())["idempotency_key"] == key
    assert not (run / "submission.rejected.json").exists()
    assert guard.submit(run, base_url=url) == guard.UNCERTAIN and len(posts) == 2


def test_initial_rejection_allows_fixed_resubmission(guard, server, run):
    url, replies, posts = server
    replies += [(422, {"title": "invalid queue"}), (200, {"id": "9.polaris"})]
    assert guard.submit(run, base_url=url) == 1
    assert not (run / "submission.started").exists()
    assert json.loads((run / "submission.rejected.json").read_text())["status"] == 422
    assert guard.submit(run, base_url=url) == 0
    assert posts[0]["headers"]["Idempotency-Key"] != posts[1]["headers"]["Idempotency-Key"]


def test_concurrent_attempts_send_one_request(guard, server, run):
    url, replies, posts = server
    replies += [(200, {"id": "5.polaris"})] * 4
    codes = []
    threads = [
        threading.Thread(target=lambda: codes.append(guard.submit(run, base_url=url)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(posts) == 1 and 0 in codes
    assert set(codes) <= {0, guard.UNCERTAIN}


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


@pytest.mark.skipif(sys.platform == "win32" or not (shutil.which("bash") and shutil.which("jq")),
                    reason="needs bash and jq")
def test_globus_record_recipe_runs(tmp_path):
    """The documented record recipe must run as written (jq reserves names like $label)."""
    cli = (SCRIPT.parents[2] / "globus-transfer/references/cli.md").read_text()
    recipe = cli.split("## Record and submit a transfer")[1].split("```bash\n")[1].split("```")[0]
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin/globus"
    fake.write_text("#!/bin/sh\necho sid-1\n")
    fake.chmod(0o755)
    values = 'STATE=.globus; LABEL=run-1-stage-in; SOURCE="L:/~/run-1/"; DESTINATION="E:/P/run-1/"\n'
    recipe = values + "\n".join(line for line in recipe.splitlines() if "=..." not in line)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(["bash", "-c", recipe], cwd=tmp_path, env=env, check=True)
    record = json.loads((tmp_path / ".globus/run-1-stage-in.json").read_text())
    assert record == {"submission_id": "sid-1", "source": "L:/~/run-1/",
                      "destination": "E:/P/run-1/", "label": "run-1-stage-in",
                      "options": ["--recursive"]}
