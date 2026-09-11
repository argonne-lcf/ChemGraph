"""Hermetic contracts for the optional web API; no model credentials needed."""

import json
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from chemgraph.api.app import create_app
from chemgraph.api.settings import Provider, Settings
from chemgraph.api.store import Store


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path, providers={"Test model": Provider(model="gpt-test")}
    )
    with TestClient(
        create_app(settings, start_workers=False),
        headers={
            "X-Auth-Request-User": "alice",
            "Origin": "http://localhost:8080",
        },
    ) as client:
        yield client


def new_session(client):
    response = client.post("/api/v1/sessions", json={"model": "Test model"})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def submit(client, session_id, **kwargs):
    return client.post(
        f"/api/v1/sessions/{session_id}/runs",
        json={
            "query": "Optimize this structure",
            "request_id": str(uuid4()),
            **kwargs,
        },
    )


def test_gateway_identity_origin_and_provider_secrets(client):
    assert client.get("/healthz").status_code == 200
    assert (
        client.get(
            "/api/v1/capabilities", headers={"X-Auth-Request-User": ""}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/sessions",
            json={"model": "Test model"},
            headers={"Origin": "https://untrusted.example"},
        ).status_code
        == 403
    )
    assert (
        client.post("/api/v1/sessions", json={"model": "unapproved"}).status_code == 422
    )
    caps = client.get("/api/v1/capabilities").json()
    assert caps["models"] == ["Test model"]
    assert "gpt-test" not in json.dumps(caps)
    assert (
        client.get("/api/v1/openapi.json").json()["info"]["title"]
        == "ChemGraph Web API"
    )


def test_every_resource_is_owned_and_downloads_are_attachments(client):
    session_id = new_session(client)
    upload = client.post(
        f"/api/v1/sessions/{session_id}/uploads?filename=../structure.xyz",
        content=b"1\ntest\nCu 0 0 0\n",
    )
    assert upload.status_code == 201
    file = upload.json()
    assert file["name"] == "structure.xyz"
    run = submit(client, session_id, attachments=[file["id"]]).json()
    paths = [
        f"/api/v1/sessions/{session_id}",
        f"/api/v1/runs/{run['id']}",
        f"/api/v1/runs/{run['id']}/events",
        file["url"],
        file["preview_url"],
    ]
    for path in paths:
        assert (
            client.get(path, headers={"X-Auth-Request-User": "bob"}).status_code == 404
        )
    assert (
        client.get("/api/v1/sessions", headers={"X-Auth-Request-User": "bob"}).json()
        == []
    )
    response = client.get(file["url"])
    assert response.content == b"1\ntest\nCu 0 0 0\n"
    assert response.headers["Content-Disposition"].startswith("attachment;")
    assert client.get(file["preview_url"]).status_code == 200
    assert "path" not in file
    assert (
        client.post(
            f"/api/v1/sessions/{session_id}/uploads?filename=x.xyz",
            content=b"x",
            headers={"X-Auth-Request-User": "bob"},
        ).status_code
        == 404
    )
    assert submit(client, session_id, query="No", attachments=[]).status_code == 409


def test_submission_idempotency_busy_sessions_and_foreign_attachments(client):
    session_id = new_session(client)
    other = new_session(client)
    file = client.post(
        f"/api/v1/sessions/{other}/uploads?filename=x.csv", content=b"x,y"
    ).json()
    assert submit(client, session_id, attachments=[file["id"]]).status_code == 404
    request_id = str(uuid4())
    response = submit(client, session_id, request_id=request_id)
    assert response.status_code == 202
    assert (
        submit(client, session_id, request_id=request_id).json()["id"]
        == response.json()["id"]
    )
    assert (
        submit(client, session_id, request_id=request_id, query="Changed").status_code
        == 409
    )
    assert submit(client, session_id).status_code == 409
    assert submit(client, other).status_code == 202


def test_missing_provider_credentials_are_reported_before_queueing(client, monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_TEST_MISSING_KEY", raising=False)
    client.app.state.settings.providers[
        "Test model"
    ].api_key_env = "CHEMGRAPH_TEST_MISSING_KEY"
    session_id = new_session(client)
    assert submit(client, session_id).status_code == 503
    assert client.get(f"/api/v1/sessions/{session_id}").json()["runs"] == []


def test_human_responses_reject_stale_blank_duplicate_and_other_users(client):
    run = submit(client, new_session(client)).json()
    question_id = str(uuid4())
    store = client.app.state.store
    store.execute(
        "UPDATE runs SET status='waiting_for_input',question_id=?,question='Proceed?' WHERE id=?",
        (question_id, run["id"]),
    )
    path = f"/api/v1/runs/{run['id']}/response"
    answer = {"question_id": question_id, "answer": "Yes"}
    assert (
        client.post(
            path, json=answer, headers={"X-Auth-Request-User": "bob"}
        ).status_code
        == 404
    )
    assert (
        client.post(path, json={**answer, "question_id": str(uuid4())}).status_code
        == 409
    )
    assert client.post(path, json={**answer, "answer": " "}).status_code == 422
    assert client.post(path, json=answer).status_code == 202
    assert client.post(path, json=answer).status_code == 409
    assert client.get(f"/api/v1/runs/{run['id']}").json()["responses"] == [
        {"question": "Proceed?", "answer": "Yes"}
    ]


def test_sse_replay_and_restart_preserve_completed_results(client):
    store = client.app.state.store
    run = submit(client, new_session(client)).json()
    store.event(run["id"], "progress", {"tool": "run_ase"})
    store.finish(run["id"], "completed", text="1 eV")
    events = store.rows(
        "SELECT id FROM events WHERE run_id=? ORDER BY id", (run["id"],)
    )
    response = client.get(
        f"/api/v1/runs/{run['id']}/events",
        headers={"Last-Event-ID": str(events[0]["id"])},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"queued"' not in response.text
    assert "run_ase" in response.text and "event: end" in response.text
    unfinished = submit(client, new_session(client)).json()
    restarted = Store(store.root)
    restarted.recover()
    assert (
        restarted.one("SELECT status FROM runs WHERE id=?", (unfinished["id"],))[
            "status"
        ]
        == "interrupted"
    )
    assert (
        restarted.one("SELECT final_text FROM runs WHERE id=?", (run["id"],))[
            "final_text"
        ]
        == "1 eV"
    )


def test_upload_limits_and_symlink_escape(client, tmp_path):
    session_id = new_session(client)
    client.app.state.settings.upload_limit = 4
    base = f"/api/v1/sessions/{session_id}/uploads"
    assert client.post(base + "?filename=x.xyz", content=b"12345").status_code == 413
    assert client.post(base + "?filename=x.xyz", content=b"").status_code == 422
    assert client.post(base + "?filename=x.py", content=b"1234").status_code == 422
    assert not list((tmp_path / "sessions").rglob("*.xyz"))
    artifact = client.post(base + "?filename=x.xyz", content=b"1234").json()
    store = client.app.state.store
    path = store.file_path(
        store.one("SELECT * FROM files WHERE id=?", (artifact["id"],))
    )
    outside = tmp_path / "private.txt"
    outside.write_text("secret")
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    assert client.get(artifact["url"]).status_code == 404
