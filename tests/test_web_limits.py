"""Admission, ownership, cancellation, provenance, and retention contracts."""

import time
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
        data_dir=tmp_path,
        providers={
            "Lab": Provider(model="test-model", base_url="http://localhost:9999/v1")
        },
    )
    with TestClient(
        create_app(settings, start_workers=False),
        headers={"Origin": settings.public_origin, "X-Auth-Request-User": "alice"},
    ) as value:
        yield value


def conversation(client, user="alice"):
    response = client.post(
        "/api/v1/sessions", json={"model": "Lab"}, headers={"X-Auth-Request-User": user}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def submit(client, session_id, user="alice", **kwargs):
    return client.post(
        f"/api/v1/sessions/{session_id}/runs",
        json={"query": "Optimize copper", "request_id": str(uuid4()), **kwargs},
        headers={"X-Auth-Request-User": user},
    )


def test_admission_limits_are_per_owner_and_retries_bypass_limits(client):
    session = conversation(client)
    request_id = str(uuid4())
    first = submit(client, session, request_id=request_id)
    assert submit(client, conversation(client)).status_code == 202
    assert submit(client, conversation(client)).status_code == 429
    assert (
        submit(client, session, request_id=request_id).json()["id"]
        == first.json()["id"]
    )
    client.app.state.settings.max_queued = 2
    assert submit(client, conversation(client, "bob"), "bob").status_code == 429
    client.post(f"/api/v1/runs/{first.json()['id']}/cancel")
    assert submit(client, conversation(client, "bob"), "bob").status_code == 202


def test_cancellation_is_owned_and_cannot_be_overwritten(client):
    session = conversation(client)
    run = submit(client, session).json()
    path = f"/api/v1/runs/{run['id']}/cancel"
    assert client.post(path, headers={"X-Auth-Request-User": "bob"}).status_code == 404
    store = client.app.state.store
    store.execute("UPDATE runs SET status='running' WHERE id=?", (run["id"],))
    assert client.post(path).json()["status"] == "cancelling"
    store.finish(run["id"], "completed", text="late result")
    assert client.get(f"/api/v1/runs/{run['id']}").json()["status"] == "cancelling"
    assert submit(client, session).status_code == 409
    store.finish(run["id"], "interrupted", stopped=True)
    assert client.post(path).json()["status"] == "cancelled"
    assert submit(client, session).status_code == 202


def test_model_change_archives_history_without_retargeting_context(client, monkeypatch):
    session = conversation(client)
    run = submit(client, session).json()
    store = client.app.state.store
    store.finish(run["id"], "completed", text="Saved answer")
    provider = client.app.state.settings.providers["Lab"]
    provider.base_url = "https://different.example/v1"
    detail = client.get(f"/api/v1/sessions/{session}").json()
    assert detail["runs"][0]["final_text"] == "Saved answer"
    assert detail["model_status"]["code"] == "model_changed"
    response = submit(client, session)
    assert response.status_code == 503
    assert response.json()["error"]["submission_rejected"]
    client.app.state.settings.providers.clear()
    assert (
        client.get(f"/api/v1/sessions/{session}").json()["model_status"]["code"]
        == "model_removed"
    )


def test_storage_quota_accounts_for_uploads_and_reservations(client):
    session = conversation(client)
    settings = client.app.state.settings
    settings.user_storage_limit = 5
    path = f"/api/v1/sessions/{session}/uploads?filename=a.txt"
    assert client.post(path, content=b"1234").status_code == 201
    assert client.post(path, content=b"56").status_code == 413
    assert submit(client, session).status_code == 413
    settings.user_storage_limit = 12
    store = client.app.state.store
    store.execute(
        "INSERT INTO upload_reservations VALUES (?,?,?)", ("reserved", session, 8)
    )
    assert client.post(path, content=b"1").status_code == 413
    store.execute("DELETE FROM upload_reservations")
    assert client.post(path, content=b"56").status_code == 201


def test_retention_removes_all_stores_but_preserves_active_and_uploading(client):
    store = client.app.state.store
    old = conversation(client)
    active = conversation(client)
    uploading = conversation(client)
    finished = submit(client, old).json()
    store.finish(finished["id"], "completed", text="Old answer")
    submit(client, active)
    store.execute(
        "INSERT INTO upload_reservations VALUES (?,?,?)", ("reserved", uploading, 1)
    )
    store.execute("UPDATE sessions SET last_activity=?", (time.time() - 40 * 86400,))
    for directory in (
        store.session_dir(old),
        store.root / "diagnostics" / old,
        store.root / "memory",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (store.session_dir(old) / "result.xyz").write_text("output")
    (store.root / "diagnostics" / old / "state.json").write_text("diagnostic")
    (store.root / "memory" / f"{old}.db").write_text("memory")
    assert store.cleanup(30) == 1
    assert client.get(f"/api/v1/sessions/{old}").status_code == 404
    assert client.get(f"/api/v1/sessions/{active}").status_code == 200
    assert client.get(f"/api/v1/sessions/{uploading}").status_code == 200
    assert not store.session_dir(old).exists()
    assert not (store.root / "diagnostics" / old).exists()
    assert not (store.root / "memory" / f"{old}.db").exists()


def test_upgrade_preserves_history_and_revokes_only_diagnostic_downloads(client):
    session = conversation(client)
    store = client.app.state.store
    store.execute("UPDATE sessions SET created='2020-01-01T00:00:00Z',last_activity=0")
    workspace = store.session_dir(session)
    workspace.mkdir(parents=True)
    diagnostic = workspace / "state_thread_legacy.json"
    diagnostic.write_text('{"raw_model_response": "private"}')
    result = workspace / "result.json"
    result.write_text('{"potential_energy": 1.0}')
    diagnostic_id = store.register_file(session, diagnostic, "data")
    result_id = store.register_file(session, result, "data")
    uploaded_id = store.register_file(session, diagnostic, "upload")

    upgraded = Store(store.root)
    migrated_activity = upgraded.one("SELECT last_activity FROM sessions")[
        "last_activity"
    ]
    assert migrated_activity >= time.time() - 10
    assert upgraded.cleanup(30) == 0
    assert client.get(f"/api/v1/artifacts/{diagnostic_id}/content").status_code == 404
    assert client.get(f"/api/v1/artifacts/{result_id}/content").status_code == 200
    assert client.get(f"/api/v1/artifacts/{uploaded_id}/content").status_code == 200
    assert diagnostic.exists()  # Retained for access-restricted diagnostics.
    assert (
        Store(store.root).one("SELECT last_activity FROM sessions")["last_activity"]
        == migrated_activity
    )
