"""Tests for the UI's Codex login status and device-code login helpers."""

import io
import time

import pytest

from chemgraph.models import codex as codex_model
from ui import codex_auth


class _FakeAccountResponse:
    def __init__(self, account):
        self._account = account

    def model_dump(self, mode=None, by_alias=None):
        return {"account": self._account, "requiresOpenaiAuth": True}


@pytest.fixture(autouse=True)
def _fresh_status_cache():
    codex_auth.invalidate_status_cache()
    yield
    codex_auth.invalidate_status_cache()


@pytest.fixture()
def sdk_and_cli(monkeypatch):
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: True)
    monkeypatch.setattr(codex_auth, "codex_cli_path", lambda: "/usr/local/bin/codex")


def test_status_reports_missing_sdk(monkeypatch):
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: False)
    status = codex_auth.account_status()
    assert status.state == codex_auth.STATE_NO_SDK
    assert not status.ready
    assert "chemgraph[codex]" in status.detail


def test_status_reports_missing_cli(monkeypatch):
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: True)
    monkeypatch.setattr(codex_auth, "codex_cli_path", lambda: None)
    status = codex_auth.account_status()
    assert status.state == codex_auth.STATE_NO_CLI
    assert not status.ready


@pytest.mark.parametrize(
    ("account", "state", "ready", "identity"),
    [
        ({"type": "chatgpt", "email": "chemist@example.com"}, "chatgpt", True, "chemist@example.com"),
        ({"type": "chatgpt"}, "chatgpt", True, None),
        ({"type": "apiKey"}, "api_key", False, None),
        (None, "logged_out", False, None),
        ({"type": "mystery"}, "logged_out", False, None),
    ],
)
def test_status_mirrors_cli_account_check(
    monkeypatch, sdk_and_cli, account, state, ready, identity
):
    monkeypatch.setattr(
        codex_model, "inspect_account", lambda: _FakeAccountResponse(account)
    )
    status = codex_auth.account_status()
    assert status.state == state
    assert status.ready is ready
    assert status.identity == identity


def test_status_is_cached_until_invalidated(monkeypatch, sdk_and_cli):
    calls = []

    def _inspect():
        calls.append(1)
        return _FakeAccountResponse({"type": "chatgpt"})

    monkeypatch.setattr(codex_model, "inspect_account", _inspect)
    assert codex_auth.account_status().ready
    assert codex_auth.account_status().ready
    assert len(calls) == 1
    codex_auth.invalidate_status_cache()
    codex_auth.account_status()
    assert len(calls) == 2
    codex_auth.account_status(use_cache=False)
    assert len(calls) == 3


def test_status_reports_sdk_failures(monkeypatch, sdk_and_cli):
    def _boom():
        raise RuntimeError("app-server exited")

    monkeypatch.setattr(codex_model, "inspect_account", _boom)
    status = codex_auth.account_status()
    assert status.state == codex_auth.STATE_ERROR
    assert "app-server exited" in status.detail


def test_validate_authentication_uses_inspect_account(monkeypatch):
    monkeypatch.setattr(
        codex_model,
        "inspect_account",
        lambda: _FakeAccountResponse({"type": "apiKey"}),
    )
    model = codex_model.CodexChatModel(model_id="gpt-5")
    with pytest.raises(codex_model.CodexAuthenticationError):
        model.validate_authentication()
    monkeypatch.setattr(
        codex_model,
        "inspect_account",
        lambda: _FakeAccountResponse({"type": "chatgpt"}),
    )
    model.validate_authentication()


def test_account_summary_never_raises():
    assert codex_model.account_summary(object()) == {"type": None, "identity": None}
    assert codex_model.account_summary({"account": {"type": "chatgpt", "id": "acc_1"}}) == {
        "type": "chatgpt",
        "identity": "acc_1",
    }


class _FakeProcess:
    """Popen stand-in whose stdout replays canned CLI output."""

    def __init__(self, text: str, returncode=None):
        self.stdout = io.StringIO(text)
        self._returncode = returncode
        self.returncode = None
        self.terminated = False

    def poll(self):
        if self._returncode is not None:
            self.returncode = self._returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self._returncode = -15


_DEVICE_OUTPUT = """Welcome to Codex [v0.145.0]
OpenAI's command-line coding agent

Follow these steps to sign in with ChatGPT using device code authorization:

1. Open this link in your browser and sign in to your account
   https://auth.openai.com/codex/device

2. Enter this one-time code (expires in 15 minutes)
   AAAAB-BB6TU

Continue only if you started this login in Codex.
"""


def test_device_login_parses_url_and_code():
    login = codex_auth.DeviceLogin(process=_FakeProcess(""))
    login.feed(_DEVICE_OUTPUT)
    assert login.url == "https://auth.openai.com/codex/device"
    assert login.code == "AAAAB-BB6TU"
    assert not login.finished
    assert not login.succeeded


def test_device_login_success_detection():
    login = codex_auth.DeviceLogin(process=_FakeProcess("", returncode=0))
    login.feed(_DEVICE_OUTPUT + "Successfully logged in\n")
    assert login.finished and login.succeeded
    failed = codex_auth.DeviceLogin(process=_FakeProcess("", returncode=1))
    failed.feed("Error: device code login is not enabled\n")
    assert failed.finished and not failed.succeeded


def test_start_device_login_runs_cli_and_reads_output(monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeProcess(_DEVICE_OUTPUT)

    monkeypatch.setattr(codex_auth, "codex_cli_path", lambda: "/opt/bin/codex")
    monkeypatch.setattr(codex_auth.subprocess, "Popen", fake_popen)
    login = codex_auth.start_device_login()
    assert calls[0][0] == ["/opt/bin/codex", "login", "--device-auth"]
    assert calls[0][1]["stdin"] is codex_auth.subprocess.DEVNULL
    deadline = time.time() + 2
    while login.code is None and time.time() < deadline:
        time.sleep(0.01)
    assert login.code == "AAAAB-BB6TU"
    login.cancel()
    assert login.process.terminated


def test_start_device_login_requires_cli(monkeypatch):
    monkeypatch.setattr(codex_auth, "codex_cli_path", lambda: None)
    with pytest.raises(RuntimeError):
        codex_auth.start_device_login()


def test_logout_runs_cli(monkeypatch):
    class _Done:
        returncode = 0
        stdout = "Successfully logged out\n"
        stderr = ""

    monkeypatch.setattr(codex_auth, "codex_cli_path", lambda: "/opt/bin/codex")
    monkeypatch.setattr(codex_auth.subprocess, "run", lambda *a, **k: _Done())
    ok, output = codex_auth.logout()
    assert ok and "logged out" in output
    monkeypatch.setattr(codex_auth, "codex_cli_path", lambda: None)
    assert codex_auth.logout()[0] is False
