"""Deployment example checks using the real MCP client and hermetic service."""

import asyncio
from copy import deepcopy
import json
import os
from types import SimpleNamespace

import httpx
import pytest

from examples.persistent_mcp import smoke_test as smoke
from tests.mcp_service_helpers import config, running_app

pytestmark = pytest.mark.skipif(os.name != "posix", reason="The service requires POSIX")


@pytest.mark.asyncio
async def test_smoke_receipt_survives_restart_and_rejects_tampering(tmp_path, monkeypatch):
    settings = config(tmp_path)
    url = settings.origin
    receipt = smoke.new_receipt(url)
    async with running_app(settings) as (app, _):
        tokens = [app.state.tokens.issue(name) for name in ("alice", "bob")]
        await smoke.exercise(url, tokens, receipt, 30, transport=httpx.ASGITransport(app=app))
        assert receipt["status"] == "complete"
        assert len({u["batch_id"] for u in receipt["users"]}) == 2
        assert len({u["input_sha256"] for u in receipt["users"]}) == 2
        assert not any(token in json.dumps(receipt) for token in tokens)
    async with running_app(settings) as (app, _):
        async def no_submission(*args, **kwargs):
            pytest.fail("Verification must never submit work")
        monkeypatch.setattr(app.state.service, "call", no_submission)
        transport = httpx.ASGITransport(app=app)
        await smoke.verify(url, tokens, receipt, transport=transport)
        changed = deepcopy(receipt)
        changed["users"][0]["updated_at"] += 1
        with pytest.raises(BaseExceptionGroup) as error:
            await smoke.verify(url, tokens, changed, transport=transport)
        assert "timestamp or artifacts changed" in smoke.describe(error.value)

    def no_network(*args, **kwargs):
        pytest.fail("Invalid receipts must fail before any network request")
    monkeypatch.setattr(smoke, "boundary_checks", no_network)
    for mutate in (
        lambda r: r.update(origin="https://foreign.example"),
        lambda r: r.update(status="incomplete"),
        lambda r: r["users"][0]["artifacts"][0].update(download_path="https://foreign.example/stolen"),
    ):
        changed = deepcopy(receipt)
        mutate(changed)
        with pytest.raises(smoke.SmokeFailure):
            await smoke.verify(url, tokens, changed)


@pytest.mark.asyncio
async def test_smoke_same_user_preflight_stops_before_calculations(tmp_path, monkeypatch):
    async with running_app(config(tmp_path)) as (app, _):
        tokens = [app.state.tokens.issue("same-user") for _ in range(2)]
        async def no_submission(*args, **kwargs):
            pytest.fail("Same-user credentials must be rejected before calculations")
        monkeypatch.setattr(app.state.service, "call", no_submission)
        receipt = smoke.new_receipt("http://localhost")
        with pytest.raises(BaseExceptionGroup) as error:
            await smoke.exercise(receipt["origin"], tokens, receipt, 5,
                                 transport=httpx.ASGITransport(app=app))
        assert "same user, or deployment isolation is broken" in smoke.describe(error.value)
        assert receipt["status"] == "incomplete"
        assert "upload" in receipt["users"][0]
        assert all("batch_id" not in user for user in receipt["users"])


@pytest.mark.asyncio
async def test_remote_reserves_receipt_before_network_and_keeps_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_ALICE", "test-alice-secret")
    monkeypatch.setenv("SMOKE_BOB", "test-bob-secret")
    args = SimpleNamespace(mode="remote", url="https://service.example", alice_token_env="SMOKE_ALICE",
                           bob_token_env="SMOKE_BOB", receipt=tmp_path / "receipt.json", timeout=5)
    async def fail_after_upload(url, tokens, receipt, timeout, save):
        assert json.loads(args.receipt.read_text())["status"] == "incomplete"
        receipt["users"][0]["upload"] = {"artifact_id": "a" * 64}
        save()
        raise smoke.SmokeFailure("interrupted")
    monkeypatch.setattr(smoke, "exercise", fail_after_upload)
    with pytest.raises(smoke.SmokeFailure, match="interrupted"):
        await smoke.remote(args, [])
    content = args.receipt.read_text()
    assert "test-alice-secret" not in content and "test-bob-secret" not in content
    assert json.loads(content)["users"][0]["upload"]["artifact_id"] == "a" * 64
    assert args.receipt.stat().st_mode & 0o777 == 0o600
    def no_network(*args, **kwargs):
        pytest.fail("Existing receipt must be rejected before network access")
    monkeypatch.setattr(smoke, "exercise", no_network)
    with pytest.raises(FileExistsError):
        await smoke.remote(args, [])
    assert args.receipt.read_text() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("crashed", [False, True])
async def test_local_unclean_shutdown_retains_data_and_does_not_restart(tmp_path, monkeypatch, crashed):
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", lambda **kwargs: str(root))
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *args, **kwargs: "test-token")
    class Process:
        returncode = None
        pid = 123456
        def terminate(self):
            pass
        async def wait(self):
            self.returncode = -9
    process = Process()
    starts = []
    async def start(*args, **kwargs):
        starts.append(args)
        return process
    monkeypatch.setattr(smoke.asyncio, "create_subprocess_exec", start)
    async def get(*args, **kwargs):
        return httpx.Response(200)
    monkeypatch.setattr(smoke.httpx.AsyncClient, "get", get)
    async def exercise(*args, **kwargs):
        if crashed:
            process.returncode = 1
    monkeypatch.setattr(smoke, "exercise", exercise)
    wait_for = asyncio.wait_for
    async def timeout_shutdown(awaitable, timeout):
        if timeout == 15:
            awaitable.close()
            raise TimeoutError
        return await wait_for(awaitable, timeout)
    monkeypatch.setattr(smoke.asyncio, "wait_for", timeout_shutdown)
    kills = []
    monkeypatch.setattr(smoke.os, "killpg", lambda *args: kills.append(args))
    with pytest.raises(smoke.SmokeFailure, match="worker groups may still be running"):
        await smoke.local(5, [])
    assert len(starts) == 1
    assert root.is_dir() and (root / "server.log").exists()
    assert bool(kills) is not crashed
