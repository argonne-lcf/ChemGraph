"""Exercise the iri-hpc submission guard against a local fake IRI server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import time

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/chemgraph/skills/iri-hpc/scripts/iri_submit.py"
)
REQUEST = {"resource_id": "polaris-id", "job_spec": {"executable": "/bin/bash"}}


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
