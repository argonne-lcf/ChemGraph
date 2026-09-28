"""Tests for the UI's Codex login status and device-code login helpers."""

import threading
import time
from types import SimpleNamespace

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
    """SDK importable; no ``codex`` on PATH is needed (bundled runtime)."""
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: True)


def test_status_reports_missing_sdk(monkeypatch):
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: False)
    status = codex_auth.account_status()
    assert status.state == codex_auth.STATE_NO_SDK
    assert not status.ready
    assert "chemgraph[codex]" in status.detail


def test_status_does_not_require_codex_on_path(monkeypatch):
    """The SDK runs its bundled runtime, so PATH is irrelevant for status."""
    import shutil

    monkeypatch.setattr(codex_auth, "sdk_available", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(
        codex_model, "inspect_account", lambda: _FakeAccountResponse({"type": "chatgpt"})
    )
    assert codex_auth.account_status().state == codex_auth.STATE_CHATGPT


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


# ---------------------------------------------------------------------------
# SDK device-code login and logout
# ---------------------------------------------------------------------------


class _FakeLoginHandle:
    def __init__(self, outcome):
        self.verification_url = "https://auth.openai.com/codex/device"
        self.user_code = "AAAAB-BB6TU"
        self.redeemed = threading.Event()
        self.outcome = outcome
        self.cancelled = False

    def wait(self):
        self.redeemed.wait(5)
        return _FakeModel(**self.outcome)

    def cancel(self):
        self.cancelled = True
        self.outcome = {"success": False, "error": "cancelled"}
        self.redeemed.set()


def _install_login_sdk(monkeypatch, outcome=None, fail_start=None):
    state = SimpleNamespace(handles=[], logouts=0, configs=[])

    class FakeConfig:
        def __init__(self, **kwargs):
            state.configs.append(kwargs)

    class FakeCodex:
        def __init__(self, config=None):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return None

        def close(self):
            state.closed = True

        def login_chatgpt_device_code(self):
            if fail_start:
                raise RuntimeError(fail_start)
            handle = _FakeLoginHandle(outcome or {"success": True, "error": None})
            state.handles.append(handle)
            return handle

        def logout(self):
            state.logouts += 1

    monkeypatch.setattr(
        codex_model, "_load_codex_sdk", lambda: (FakeCodex, FakeConfig, object, object)
    )
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: True)
    return state


def test_device_login_publishes_code_and_completes(monkeypatch):
    state = _install_login_sdk(monkeypatch)
    login = codex_auth.start_device_login()
    assert login.url == "https://auth.openai.com/codex/device"
    assert login.code == "AAAAB-BB6TU"
    assert not login.finished and not login.succeeded
    # Same isolated session as model calls: API keys cleared.
    assert state.configs[0]["env"] == {"OPENAI_API_KEY": "", "CODEX_API_KEY": ""}

    state.handles[0].redeemed.set()
    assert login.wait(timeout=5)
    assert login.succeeded and login.output == ""


def test_device_login_reports_failure_and_cancel(monkeypatch):
    state = _install_login_sdk(monkeypatch, outcome={"success": False, "error": "code expired"})
    login = codex_auth.start_device_login()
    state.handles[0].redeemed.set()
    assert login.wait(timeout=5)
    assert not login.succeeded and login.output == "code expired"

    state = _install_login_sdk(monkeypatch)
    login = codex_auth.start_device_login()
    login.cancel()
    assert state.handles[0].cancelled
    assert login.wait(timeout=5)
    assert login.cancelled and not login.succeeded


def test_start_device_login_surfaces_sdk_errors(monkeypatch):
    _install_login_sdk(monkeypatch, fail_start="app-server exited")
    with pytest.raises(RuntimeError, match="app-server exited"):
        codex_auth.start_device_login()
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: False)
    with pytest.raises(RuntimeError, match="not installed"):
        codex_auth.start_device_login()


def test_logout_uses_sdk_and_invalidates_status(monkeypatch):
    state = _install_login_sdk(monkeypatch)
    codex_auth._status_cache["value"] = codex_auth.CodexStatus(codex_auth.STATE_CHATGPT, "x")
    codex_auth._status_cache["at"] = time.time()
    ok, message = codex_auth.logout()
    assert ok and state.logouts == 1
    assert codex_auth._status_cache["value"] is None

    def _boom():
        raise RuntimeError("no runtime")

    monkeypatch.setattr(codex_model, "logout_account", _boom)
    assert codex_auth.logout() == (False, "no runtime")


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self, mode=None, by_alias=None):
        return dict(self._fields)


class _FakeModelList:
    def __init__(self, models):
        self._models = models

    def model_dump(self, mode=None, by_alias=None):
        return {"data": [m.model_dump() for m in self._models], "nextCursor": None}


_CATALOG = [
    _FakeModel(id="gpt-5.1-codex", model="gpt-5.1-codex", displayName="GPT-5.1 Codex",
               description="Coding model", isDefault=True, hidden=False),
    _FakeModel(id="gpt-5.1", model="gpt-5.1", displayName="GPT-5.1",
               description="General", isDefault=False, hidden=False),
    _FakeModel(id="secret", model="secret-preview", displayName="Preview",
               description="", isDefault=False, hidden=True),
    _FakeModel(id="legacy-id-only", displayName="Legacy", isDefault=False, hidden=False),
    _FakeModel(displayName="No identifier", isDefault=False, hidden=False),
]


def _install_fake_sdk(monkeypatch, catalog=_CATALOG, calls=None):
    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCodex:
        def __init__(self, config=None):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return None

        def models(self, include_hidden=False):
            if calls is not None:
                calls.append((self.config.kwargs, include_hidden))
            return _FakeModelList(catalog)

    monkeypatch.setattr(
        codex_model, "_load_codex_sdk", lambda: (FakeCodex, FakeConfig, object, object)
    )


def test_list_models_normalizes_sdk_catalog(monkeypatch):
    calls = []
    _install_fake_sdk(monkeypatch, calls=calls)
    models = codex_model.list_models()
    assert [m["model"] for m in models] == [
        "gpt-5.1-codex", "gpt-5.1", "secret-preview", "legacy-id-only",
    ]
    assert models[0] == {
        "model": "gpt-5.1-codex", "display_name": "GPT-5.1 Codex",
        "description": "Coding model", "is_default": True, "hidden": False,
    }
    assert models[2]["hidden"] is True
    # Same isolation as the account check: cleared API keys, throwaway cwd.
    assert calls[0][0]["env"] == {"OPENAI_API_KEY": "", "CODEX_API_KEY": ""}
    assert calls[0][1] is False


def test_available_models_requires_login_and_hides_hidden(monkeypatch, sdk_and_cli):
    _install_fake_sdk(monkeypatch)
    monkeypatch.setattr(
        codex_model, "inspect_account", lambda: _FakeAccountResponse({"type": "apiKey"})
    )
    assert codex_auth.available_models() == []
    assert codex_auth.default_model_name() is None

    codex_auth.invalidate_status_cache()
    monkeypatch.setattr(
        codex_model, "inspect_account", lambda: _FakeAccountResponse({"type": "chatgpt"})
    )
    models = codex_auth.available_models()
    assert [m["name"] for m in models] == [
        "codex:gpt-5.1-codex", "codex:gpt-5.1", "codex:legacy-id-only",
    ]
    assert models[0]["display_name"] == "GPT-5.1 Codex" and models[0]["is_default"]
    assert codex_auth.default_model_name() == "codex:gpt-5.1-codex"
    assert codex_auth.default_model_name(models[1:]) == "codex:gpt-5.1"


def test_available_models_is_cached_and_survives_sdk_failure(monkeypatch, sdk_and_cli):
    monkeypatch.setattr(
        codex_model, "inspect_account", lambda: _FakeAccountResponse({"type": "chatgpt"})
    )
    calls = []
    _install_fake_sdk(monkeypatch, calls=calls)
    codex_auth.available_models()
    codex_auth.available_models()
    assert len(calls) == 1
    codex_auth.invalidate_status_cache()

    def _boom():
        raise RuntimeError("app-server down")

    monkeypatch.setattr(codex_model, "list_models", _boom)
    assert codex_auth.available_models() == []


def _install_slow_login_sdk(monkeypatch, *, issue_delay_event, wait_forever=True):
    state = SimpleNamespace(closed=False, handle=None, waiting=threading.Event())

    class Handle:
        verification_url = "https://auth.openai.com/codex/device"
        user_code = "AAAAB-BB6TU"

        def __init__(self, codex):
            self.codex = codex
            self.cancelled = False

        def wait(self):
            state.waiting.set()
            # Like the SDK: blocks until a notification or the session closes.
            while not self.codex.closed.is_set():
                self.codex.closed.wait(0.05)
            raise RuntimeError("transport closed")

        def cancel(self):
            self.cancelled = True

    class FakeCodex:
        def __init__(self, config=None):
            self.closed = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            self.close()

        def close(self):
            state.closed = True
            self.closed.set()

        def login_chatgpt_device_code(self):
            # Code issuance is slow; closing the session aborts the request.
            while not issue_delay_event.is_set():
                if self.closed.wait(0.05):
                    raise RuntimeError("transport closed")
            state.handle = Handle(self)
            return state.handle

    class FakeConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        codex_model, "_load_codex_sdk", lambda: (FakeCodex, FakeConfig, object, object)
    )
    monkeypatch.setattr(codex_auth, "sdk_available", lambda: True)
    return state


def test_start_timeout_cancels_and_releases_the_session(monkeypatch):
    issued = threading.Event()
    state = _install_slow_login_sdk(monkeypatch, issue_delay_event=issued)
    monkeypatch.setattr(codex_auth, "LOGIN_START_TIMEOUT", 0.2)
    with pytest.raises(RuntimeError, match="did not issue a sign-in code"):
        codex_auth.start_device_login()
    # The pending code request was aborted by closing the session: the
    # thread never went on to wait for a login nobody will complete.
    deadline = time.time() + 5
    while not state.closed and time.time() < deadline:
        time.sleep(0.02)
    assert state.closed
    assert not state.waiting.is_set()


def test_code_arriving_after_cancel_is_cancelled_not_awaited(monkeypatch):
    issued = threading.Event()
    state = _install_slow_login_sdk(monkeypatch, issue_delay_event=issued)
    login = codex_auth.DeviceLogin()
    login.start(timeout=0.1)
    with login._lock:
        login.cancelled = True
        login._codex = None  # cancellation before the session was published
    issued.set()
    assert login.wait(timeout=5)
    assert state.handle is not None and state.handle.cancelled
    assert not state.waiting.is_set()
    assert login.cancelled and not login.succeeded and login.code is None


def test_cancel_while_waiting_ends_the_thread(monkeypatch):
    issued = threading.Event()
    issued.set()
    state = _install_slow_login_sdk(monkeypatch, issue_delay_event=issued)
    login = codex_auth.start_device_login()
    assert state.waiting.wait(5)
    login.cancel()
    assert login.wait(timeout=5)
    assert state.handle.cancelled and state.closed
    assert login.cancelled and not login.succeeded and login.output == ""
