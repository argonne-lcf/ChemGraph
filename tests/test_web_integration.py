"""Real web workers/graphs against a deterministic loopback model endpoint."""

from http.server import ThreadingHTTPServer
import json
import threading
import time
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fcntl")
from fastapi.testclient import TestClient

from chemgraph.api.app import create_app
from chemgraph.api.settings import Provider, Settings
from chemgraph.api.memory import WebSessionStore
from tests.web_model_server import Handler


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-only-placeholder")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def wait_for(client, run_id, states):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in states:
            return run
        time.sleep(0.05)
    pytest.fail(f"Run did not reach {states}: {run}")


@pytest.mark.parametrize("workflow", ["single_agent", "multi_agent"])
def test_real_graph_emt_clarification_memory_and_artifacts(
    tmp_path, endpoint, workflow
):
    settings = Settings(
        data_dir=tmp_path,
        dev_user="test",
        providers={"Lab": Provider(model="test-model", base_url=endpoint)},
    )
    with TestClient(
        create_app(settings), headers={"Origin": settings.public_origin}
    ) as client:
        session = client.post(
            "/api/v1/sessions", json={"model": "Lab", "workflow": workflow}
        ).json()["id"]
        attachment = client.post(
            f"/api/v1/sessions/{session}/uploads?filename=copper.xyz",
            content=b"2\nCopper\nCu 0 0 0\nCu 0 0 2.5\n",
        ).json()
        first_query = "Confirm optimization of the attached copper with EMT"
        run = client.post(
            f"/api/v1/sessions/{session}/runs",
            json={
                "query": first_query,
                "attachments": [attachment["id"]],
                "request_id": str(uuid4()),
            },
        ).json()
        waiting = wait_for(client, run["id"], {"waiting_for_input", "failed"})
        assert waiting["status"] == "waiting_for_input", waiting
        client.post(
            f"/api/v1/runs/{run['id']}/response",
            json={"question_id": waiting["question_id"], "answer": "Yes, continue"},
        )
        finished = wait_for(client, run["id"], {"completed", "failed", "interrupted"})
        assert finished["status"] == "completed", finished
        assert "EMT potential energy" in finished["final_text"]
        artifacts = {item["name"]: item for item in finished["artifacts"]}
        assert set(artifacts) == {"copper-result.json", "copper_opt.traj"}
        result = client.get(artifacts["copper-result.json"]["url"]).json()
        assert result["potential_energy"] > 0
        assert (
            client.get(artifacts["copper_opt.traj"]["preview_url"]).text.count("Cu") > 2
        )
        events = client.get(f"/api/v1/runs/{run['id']}/events").text
        assert "run_ase" in events and "tool_call_finished" in events
        assert list((tmp_path / "diagnostics" / session).rglob("state_thread_*.json"))
        for index in range(3):
            followup = client.post(
                f"/api/v1/sessions/{session}/runs",
                json={
                    "query": f"Follow-up: recall our work ({index})",
                    "request_id": str(uuid4()),
                },
            ).json()
            answer = wait_for(client, followup["id"], {"completed", "failed"})
            assert "Previous context retained" in answer["final_text"], answer
        memory = WebSessionStore(
            str(tmp_path / "memory" / f"{session}.db"), "unused", 1000
        )
        messages = memory.get_session(session).messages
        humans = [message.content for message in messages if message.role == "human"]
        assert humans[0] == first_query
        assert all(
            "Previous Session" not in content and "Now, continuing" not in content
            for content in humans
        )
        assert len(memory.build_context_summary(session)) <= 1000


@pytest.mark.parametrize(
    "status,code", [(401, "provider_authentication"), (429, "provider_rate_limit")]
)
def test_provider_failures_do_not_expose_raw_responses(
    tmp_path, endpoint, status, code
):
    settings = Settings(
        data_dir=tmp_path,
        dev_user="test",
        providers={"Lab": Provider(model=f"fail-{status}", base_url=endpoint)},
    )
    with TestClient(
        create_app(settings), headers={"Origin": settings.public_origin}
    ) as client:
        session = client.post("/api/v1/sessions", json={"model": "Lab"}).json()["id"]
        run = client.post(
            f"/api/v1/sessions/{session}/runs",
            json={"query": "test provider failure", "request_id": str(uuid4())},
        ).json()
        failed = wait_for(client, run["id"], {"failed", "interrupted"})
        assert failed["error_code"] == code, failed
        assert "PRIVATE_PROVIDER_RESPONSE_MARKER" not in json.dumps(failed)
        assert not failed["artifacts"]


def test_waiting_owner_does_not_starve_other_users_and_can_cancel(tmp_path):
    settings = Settings(data_dir=tmp_path, demo=True, dev_user="alice")
    with TestClient(
        create_app(settings), headers={"Origin": settings.public_origin}
    ) as client:

        def start():
            session = client.post("/api/v1/sessions", json={"model": "demo"}).json()[
                "id"
            ]
            return client.post(
                f"/api/v1/sessions/{session}/runs",
                json={"query": "Confirm", "request_id": str(uuid4())},
            ).json()["id"]

        alice = start()
        wait_for(client, alice, {"waiting_for_input"})
        queued = start()
        settings.dev_user = "bob"
        bob = start()
        wait_for(client, bob, {"waiting_for_input"})
        settings.dev_user = "alice"
        assert client.get(f"/api/v1/runs/{queued}").json()["status"] == "queued"
        client.post(f"/api/v1/runs/{alice}/cancel")
        assert wait_for(client, alice, {"cancelled"})["error_code"] == "cancelled"
        wait_for(client, queued, {"waiting_for_input"})


def test_clarification_timeout_releases_worker(tmp_path):
    settings = Settings(
        data_dir=tmp_path, demo=True, dev_user="test", clarification_timeout=1
    )
    with TestClient(
        create_app(settings), headers={"Origin": settings.public_origin}
    ) as client:
        session = client.post("/api/v1/sessions", json={"model": "demo"}).json()["id"]
        run = client.post(
            f"/api/v1/sessions/{session}/runs",
            json={"query": "Confirm", "request_id": str(uuid4())},
        ).json()
        failed = wait_for(client, run["id"], {"interrupted"})
        assert failed["error_code"] == "clarification_timeout"


@pytest.mark.llm
def test_enabled_live_providers_smoke(tmp_path):
    """Operator opt-in: use the real configured providers with EMT only."""
    settings = Settings.from_env().model_copy(
        update={
            "data_dir": tmp_path,
            "dev_user": "live-smoke",
            "demo": False,
            "calculators": ("emt",),
            "provider_timeout": 30,
            "run_timeout": 180,
        }
    )
    assert settings.providers, (
        "Configure CHEMGRAPH_WEB_PROVIDERS_FILE or CHEMGRAPH_WEB_PROVIDERS first."
    )
    with TestClient(
        create_app(settings), headers={"Origin": settings.public_origin}
    ) as client:
        for label in settings.providers:
            session_response = client.post("/api/v1/sessions", json={"model": label})
            assert session_response.status_code == 201, session_response.text
            session = session_response.json()["id"]
            attachment = client.post(
                f"/api/v1/sessions/{session}/uploads?filename=copper.xyz",
                content=b"2\nCopper\nCu 0 0 0\nCu 0 0 2.5\n",
            ).json()
            run = client.post(
                f"/api/v1/sessions/{session}/runs",
                json={
                    "query": "Use run_ase with EMT to calculate the single-point potential energy of the attached copper. Save results to copper.json and report the energy in eV.",
                    "attachments": [attachment["id"]],
                    "request_id": str(uuid4()),
                },
            ).json()
            deadline = time.monotonic() + 190
            while time.monotonic() < deadline:
                result = client.get(f"/api/v1/runs/{run['id']}").json()
                if result["status"] == "waiting_for_input":
                    client.post(
                        f"/api/v1/runs/{run['id']}/response",
                        json={
                            "question_id": result["question_id"],
                            "answer": "Proceed with the attached copper, EMT, single-point energy.",
                        },
                    )
                if result["status"] in {"completed", "failed", "interrupted"}:
                    break
                time.sleep(0.5)
            assert result["status"] == "completed", result
            artifacts = [
                item for item in result["artifacts"] if item["name"].endswith(".json")
            ]
            assert artifacts, "The provider must actually call the chemistry tool."
            assert any(
                client.get(item["url"]).json().get("potential_energy", 0) > 0
                for item in artifacts
            )
