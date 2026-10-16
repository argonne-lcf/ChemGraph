import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chemgraph.distributed.cli import process_controls, queue_control
from chemgraph.distributed.config import DistributedConfig, load_config
from chemgraph.observability.run_files import write_json_atomic


def config(tmp_path):
    return DistributedConfig(run_id="run", launch_id="launch", exchange_url="http://unused", run_dir=tmp_path,
                             peers={"worker": {}})


@pytest.mark.asyncio
async def test_explicit_local_operator_resume(tmp_path):
    settings = config(tmp_path)
    status = {"state": "waiting_for_user", "current_message_id": "message", "instance_id": "instance",
              "interrupts": [{"id": "approve"}]}
    write_json_atomic(settings.peer_dir("worker") / "status.json", status)
    response = {"decisions": [{"type": "reject"}]}
    path = queue_control(settings, "worker", "resume", response)
    runtime = SimpleNamespace(status=lambda: status, instance_id="instance", resume=AsyncMock())
    await process_controls(settings.peer_dir("worker"), runtime, asyncio.Event())
    runtime.resume.assert_awaited_once_with(response)
    assert json.loads(path.read_text()) == {"status": "completed"}


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["message", "process", "interrupt"])
async def test_stale_approval_never_resumes_new_work(tmp_path, changed):
    settings = config(tmp_path)
    status = {"state": "waiting_for_user", "current_message_id": "message", "instance_id": "instance", "interrupts": []}
    write_json_atomic(settings.peer_dir("worker") / "status.json", status)
    path = queue_control(settings, "worker", "resume", "approve")
    runtime = SimpleNamespace(status=lambda: status, instance_id="instance", resume=AsyncMock())
    if changed == "process":
        runtime.instance_id = "new"
    elif changed == "message":
        status["current_message_id"] = "new"
    else:
        status["interrupts"] = [{"id": "new"}]
    await process_controls(settings.peer_dir("worker"), runtime, asyncio.Event())
    runtime.resume.assert_not_called()
    assert json.loads(path.read_text())["status"] == "failed"


@pytest.mark.asyncio
async def test_shutdown_does_not_wait_for_operator_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("chemgraph.distributed.cli.uuid", SimpleNamespace(uuid4=iter(["a", "z"]).__next__))
    settings = config(tmp_path)
    directory = settings.peer_dir("worker")
    status = {"state": "waiting_for_user", "current_message_id": "message",
              "instance_id": "instance", "interrupts": []}
    write_json_atomic(directory / "status.json", status)
    queue_control(settings, "worker", "resume", "approve")
    path = queue_control(settings, "worker", "shutdown")
    resumed = asyncio.Event()
    finish = asyncio.Event()

    async def resume(_response):
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
