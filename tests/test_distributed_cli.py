import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from chemgraph.distributed.cli import _transport, process_controls, queue_control, run_peer
from chemgraph.distributed.config import DistributedConfig, load_config
from chemgraph.distributed.protocol import PeerEnvelope
from chemgraph.distributed.runner import ChemGraphRunner
from chemgraph.distributed.runtime import PeerRuntime
from chemgraph.observability.event_log import read_events
from chemgraph.observability.run_files import write_json_atomic


def config(tmp_path):
    return DistributedConfig(run_id="run", launch_id="launch", exchange_url="http://unused", run_dir=tmp_path,
                             peers={"worker": {}})


@pytest.mark.asyncio
async def test_explicit_local_operator_resume(tmp_path):
    settings = config(tmp_path)
    status = {"state": "waiting_for_user", "current_message_id": "message", "instance_id": "instance",
              "control_token": "token", "interrupts": [{"id": "approve"}]}
    write_json_atomic(settings.peer_dir("worker") / "status.json", status)
    response = {"decisions": [{"type": "reject"}]}
    path = queue_control(settings, "worker", "resume", response)
    runtime = SimpleNamespace(status=lambda: status, instance_id="instance", resume=AsyncMock())
    await process_controls(settings.peer_dir("worker"), runtime, asyncio.Event())
    runtime.resume.assert_awaited_once_with(response, control_token="token")
    assert json.loads(path.read_text()) == {"status": "completed"}


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["message", "process", "interrupt"])
async def test_stale_approval_never_resumes_new_work(tmp_path, changed):
    settings = config(tmp_path)
    status = {"state": "waiting_for_user", "current_message_id": "message", "instance_id": "instance",
              "control_token": "token", "interrupts": []}
    write_json_atomic(settings.peer_dir("worker") / "status.json", status)
    path = queue_control(settings, "worker", "resume", "approve")
    runtime = SimpleNamespace(status=lambda: status, instance_id="instance", resume=AsyncMock())
    if changed == "process":
        runtime.instance_id = "new"
    elif changed == "message":
        status["current_message_id"] = "new"
    else:
        status["control_token"] = "new"
    await process_controls(settings.peer_dir("worker"), runtime, asyncio.Event())
    runtime.resume.assert_not_called()
    assert json.loads(path.read_text())["status"] == "failed"


@pytest.mark.asyncio
async def test_shutdown_does_not_wait_for_operator_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("chemgraph.distributed.cli.uuid", SimpleNamespace(uuid4=iter(["a", "z"]).__next__))
    settings = config(tmp_path)
    directory = settings.peer_dir("worker")
    status = {"state": "waiting_for_user", "current_message_id": "message",
              "instance_id": "instance", "control_token": "token", "interrupts": []}
    write_json_atomic(directory / "status.json", status)
    queue_control(settings, "worker", "resume", "approve")
    path = queue_control(settings, "worker", "shutdown")
    resumed = asyncio.Event()
    finish = asyncio.Event()

    async def resume(_response, *, control_token):
        resumed.set()
        await finish.wait()

    runtime = SimpleNamespace(status=lambda: status, instance_id="instance", resume=resume)
    stop = asyncio.Event()
    task = asyncio.create_task(process_controls(directory, runtime, stop))
    try:
        await asyncio.wait_for(resumed.wait(), 1)
        await process_controls(directory, runtime, stop, shutdown_only=True)
        assert stop.is_set()
        assert not task.done()
        assert json.loads(path.read_text()) == {"status": "completed"}
    finally:
        finish.set()
        await task
    assert json.loads(path.read_text()) == {"status": "completed"}


def test_config_resolves_local_paths_and_validates_names(tmp_path):
    path = tmp_path / "peers.json"
    path.write_text(json.dumps({"run_id": "run", "launch_id": "launch", "exchange_url": "http://unused", "peers": {"one": {}}}))
    loaded = load_config(path)
    assert loaded.peer_dir("one") == tmp_path / "distributed_runs/run/launch/one"
    assert loaded.peers["one"].workflow_type == "main_agent"
    for peers in ({"../escape": {}}, {"one": {"allowed_peers": ["missing"]}}, {"operator": {}}):
        with pytest.raises(ValueError):
            DistributedConfig(run_id="run", launch_id="launch", exchange_url="http://unused", peers=peers)
    with pytest.raises(ValueError, match="runtime-owned fields"):
        DistributedConfig(run_id="run", launch_id="launch", exchange_url="http://unused",
                          peers={"one": {"options": {"tool_registry": None}}})


def test_distributed_help_forwarding(monkeypatch, capsys):
    from chemgraph.cli.main import main
    monkeypatch.setattr(sys, "argv", ["chemgraph", "distributed", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "resume" in capsys.readouterr().out


@pytest.mark.parametrize("method", [None, "globus"])
def test_exchange_authentication_forwarding(tmp_path, monkeypatch, method):
    pytest.importorskip("academy")
    factory = Mock()
    monkeypatch.setattr("academy.exchange.cloud.HttpExchangeFactory", factory)
    monkeypatch.setenv("TEST_EXCHANGE_AUTH", "Bearer dummy-test-token")
    settings = config(tmp_path)
    settings.exchange_auth_method = method
    settings.exchange_header_env = {"authorization": "TEST_EXCHANGE_AUTH"}
    _transport(settings)
    factory.assert_called_once_with(url=settings.exchange_url, auth_method=method,
                                    additional_headers={"Authorization": "Bearer dummy-test-token"})


def test_missing_exchange_header_fails_before_connecting(tmp_path, monkeypatch):
    pytest.importorskip("academy")
    factory = Mock()
    monkeypatch.setattr("academy.exchange.cloud.HttpExchangeFactory", factory)
    monkeypatch.delenv("TEST_EXCHANGE_AUTH", raising=False)
    settings = config(tmp_path)
    settings.exchange_header_env = {"Authorization": "TEST_EXCHANGE_AUTH"}
    with pytest.raises(ValueError, match="requires environment variable 'TEST_EXCHANGE_AUTH'"):
        _transport(settings)
    factory.assert_not_called()


def test_exchange_authentication_is_not_checkpoint_identity(tmp_path):
    from chemgraph.distributed.runner import _configuration_id
    settings = config(tmp_path)
    original = _configuration_id(settings, "worker")
    settings.exchange_auth_method = "globus"
    settings.exchange_header_env = {"Authorization": "TEST_EXCHANGE_AUTH"}
    assert _configuration_id(settings, "worker") == original


@pytest.mark.asyncio
async def test_non_json_native_interrupts_round_trip_through_operator_control(tmp_path):
    payload = {"choices": ("EMT", "MACE"), "path": tmp_path, "custom": SimpleNamespace(value=1)}
    class Agent:
        log_dir = str(tmp_path)
        async def run(self, prompt, config):
            return await self.human_input_handler("Choose", payload)

    settings = config(tmp_path)
    runner = ChemGraphRunner(Agent(), thread_id="legacy")
    peer = PeerRuntime(name="worker", run_id="run", launch_id="launch", allowed_senders=(), runner=runner)
    await peer.start()
    try:
        await peer.receive(PeerEnvelope(run_id="run", launch_id="launch", sender="operator",
                                        recipient="worker", content="task"))
        async with asyncio.timeout(2):
            while peer.status()["state"] != "waiting_for_user":
                await asyncio.sleep(0)
        status = peer.status()
        assert status["interrupts"][0]["payload"] == {
            "choices": ["EMT", "MACE"], "path": str(tmp_path), "custom": "namespace(value=1)",
        }
        assert runner.interrupts[0]["payload"] is payload
        write_json_atomic(settings.peer_dir("worker") / "status.json", status)
        result = queue_control(settings, "worker", "resume", "EMT")
        await process_controls(settings.peer_dir("worker"), peer, asyncio.Event())
        assert json.loads(result.read_text()) == {"status": "completed"}
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert peer.output == "EMT"
    finally:
        await peer.close(timeout_s=0.01)


@pytest.mark.skipif(sys.platform == "win32", reason="Peer launcher requires POSIX file locking")
@pytest.mark.asyncio
async def test_peer_startup_records_workflow_and_host(tmp_path, monkeypatch):
    @asynccontextmanager
    async def transport(*_args):
        yield SimpleNamespace(start=AsyncMock(), check_health=lambda: None)

    @asynccontextmanager
    async def runner(*_args):
        yield SimpleNamespace(interrupts=())

    monkeypatch.setattr("chemgraph.distributed.cli._transport", transport)
    monkeypatch.setattr("chemgraph.distributed.cli.ConfiguredRunner", runner)
    monkeypatch.setattr("chemgraph.distributed.cli.socket.gethostname", lambda: "compute.example")
    settings = config(tmp_path)
    settings.peers["worker"].idle_timeout_s = 0.001
    await run_peer(settings, "worker")
    events = read_events(settings.peer_dir("worker") / "events.jsonl")
    started = next(event for event in events if event.event == "agent_started")
    assert started.payload == {"workflow_type": "main_agent", "placement": {
        "hostname": "compute.example", "short_hostname": "compute",
    }}
