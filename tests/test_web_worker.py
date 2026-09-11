"""Exercise real process scheduling and a mocked model through the web worker."""

import asyncio
import time
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fcntl")

from fastapi.testclient import TestClient
from chemgraph.api.app import create_app
from chemgraph.api.settings import Provider, Settings
from chemgraph.api.store import Store
from chemgraph.api.worker import execute_run, Supervisor


def wait_for(client, run_id, expected):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row = client.get(f"/api/v1/runs/{run_id}").json()
        if row["status"] in expected:
            return row
        time.sleep(0.1)
    pytest.fail(f"Run {run_id} did not reach {expected}: {row}")


def test_processes_emt_artifacts_and_human_response(tmp_path):
    settings = Settings(data_dir=tmp_path, demo=True, dev_user="local-demo")
    with TestClient(
        create_app(settings), headers={"Origin": settings.public_origin}
    ) as client:
        ids = []
        for query in ("Optimize copper", "Confirm optimization"):
            session = client.post("/api/v1/sessions", json={"model": "demo"}).json()
            result = client.post(
                f"/api/v1/sessions/{session['id']}/runs",
                json={"query": query, "request_id": str(uuid4())},
            )
            assert result.status_code == 202
            ids.append(result.json()["id"])
        first = wait_for(client, ids[0], {"completed", "failed"})
        assert first["status"] == "completed", first
        waiting = wait_for(client, ids[1], {"waiting_for_input", "failed"})
        assert waiting["status"] == "waiting_for_input", waiting
        assert (
            client.post(
                f"/api/v1/runs/{ids[1]}/response",
                json={"question_id": waiting["question_id"], "answer": "Yes"},
            ).status_code
            == 202
        )
        second = wait_for(client, ids[1], {"completed", "failed"})
        assert second["status"] == "completed", second
        first_files = {item["name"]: item for item in first["artifacts"]}
        second_files = {item["name"]: item for item in second["artifacts"]}
        assert "copper_opt.traj" in first_files
        assert first_files["copper.xyz"]["id"] != second_files["copper.xyz"]["id"]
        assert client.get(first_files["copper.xyz"]["preview_url"]).status_code == 200
        assert (
            client.get(first_files["copper_opt.traj"]["preview_url"]).text.count("Cu")
            >= 4
        )
        assert str(tmp_path) not in first["final_text"]
        assert client.get(f"/api/v1/runs/{ids[0]}/events").text.count("event: end") == 1


def test_server_rejects_second_supervisor(tmp_path):
    async def exercise():
        settings = Settings(data_dir=tmp_path)
        one = Supervisor(Store(tmp_path), settings)
        two = Supervisor(Store(tmp_path), settings)
        await one.start()
        try:
            with pytest.raises(RuntimeError, match="one API process"):
                await two.start()
        finally:
            await one.stop()

    asyncio.run(exercise())


def test_worker_reuses_memory_and_does_not_forward_secret_events(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from chemgraph.agent import llm_agent
    from chemgraph.memory.schemas import SessionMessage

    settings = Settings(
        data_dir=tmp_path,
        providers={"test": Provider(model="test-model", api_key_env="WEB_TEST_KEY")},
    )
    store = Store(tmp_path)
    session_id = str(uuid4())
    store.execute(
        "INSERT INTO sessions(id,owner,model,workflow,title) VALUES (?,?,?,?,?)",
        (session_id, "owner", "test", "single_agent", "Test"),
    )
    monkeypatch.setenv("WEB_TEST_KEY", "private-test-key")
    # execute_run changes only the child environment in production. Restore the
    # cwd/env here because this mocked test calls the entry point in-process.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    seen = []

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, query, resume_from=None):
            assert self.kwargs["api_key"] == "private-test-key"
            seen.append(resume_from)
            self.kwargs["on_event"](
                "tool_call_started",
                {"tool_name": "run_ase", "arguments": "private-test-key"},
            )
            memory = self.kwargs["session_store"]
            if not memory.get_session(self.uuid):
                memory.create_session(self.uuid, "test-model", "single_agent")
            memory.save_messages(
                self.uuid,
                [
                    SessionMessage(role="human", content=query),
                    SessionMessage(role="ai", content="Done"),
                ],
            )
            return AIMessage(content="Done")

    monkeypatch.setattr(llm_agent, "ChemGraph", Agent)
    for _ in range(2):
        run_id = str(uuid4())
        store.execute(
            "INSERT INTO runs(id,session_id,query,request_id) VALUES (?,?,?,?)",
            (run_id, session_id, "Question", str(uuid4())),
        )
        execute_run(str(tmp_path), run_id, settings.model_dump())
        assert (
            store.one("SELECT status FROM runs WHERE id=?", (run_id,))["status"]
            == "completed"
        )
    assert seen == [None, session_id]
    assert "private-test-key" not in str(store.rows("SELECT * FROM events"))


def test_shutdown_interrupts_waiting_run(tmp_path):
    settings = Settings(data_dir=tmp_path, dev_user="demo", demo=True)
    with TestClient(
        create_app(settings), headers={"Origin": settings.public_origin}
    ) as client:
        session_id = client.post("/api/v1/sessions", json={"model": "demo"}).json()[
            "id"
        ]
        run_id = client.post(
            f"/api/v1/sessions/{session_id}/runs",
            json={"query": "Confirm", "request_id": str(uuid4())},
        ).json()["id"]
        wait_for(client, run_id, {"waiting_for_input"})
    assert (
        Store(tmp_path).one("SELECT status FROM runs WHERE id=?", (run_id,))["status"]
        == "interrupted"
    )
