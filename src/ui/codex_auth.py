"""Codex (ChatGPT subscription) login status and device-code login for the UI.

The CLI authorizes ``codex:<model-id>`` models through the login stored by
Codex; ChemGraph then validates that the active login is ChatGPT-managed
rather than an API key.  This module exposes the same checks to the
Streamlit UI and drives login/logout through the pinned ``openai-codex``
SDK, which runs its own bundled Codex runtime: a device-code login
(``Codex.login_chatgpt_device_code``) yields a verification URL and a
one-time code that the page shows, and a background thread waits for the
SDK's completion notification.  No ``codex`` executable on ``PATH`` is
needed.

Streamlit-free so it can be unit-tested with a mocked SDK.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from chemgraph.models.codex import CODEX_MODEL_PREFIX

#: Login states reported by :func:`account_status`.
STATE_NO_SDK = "no_sdk"
STATE_LOGGED_OUT = "logged_out"
STATE_API_KEY = "api_key"
STATE_CHATGPT = "chatgpt"
STATE_ERROR = "error"

READY_STATES = frozenset({STATE_CHATGPT})

INSTALL_HINT = (
    "Install the optional `chemgraph[codex]` extra (it bundles the pinned "
    "Codex runtime), then sign in with ChatGPT authentication."
)

#: Seconds :func:`start_device_login` waits for the SDK to issue a code.
LOGIN_START_TIMEOUT = 30.0


@dataclass(frozen=True)
class CodexStatus:
    """Readiness of the Codex subscription route."""

    state: str
    detail: str
    identity: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.state in READY_STATES


def sdk_available() -> bool:
    """Return whether the optional ``openai-codex`` SDK is importable."""
    import importlib.util

    return importlib.util.find_spec("openai_codex") is not None


#: Seconds a status lookup stays valid.  Querying the login spawns the Codex
#: app-server, which is too slow to repeat on every Streamlit rerun.
STATUS_CACHE_TTL = 30.0
_status_cache: dict[str, object] = {"at": 0.0, "value": None}


_models_cache: dict[str, object] = {"at": 0.0, "value": None}


def invalidate_status_cache() -> None:
    """Forget the cached login status and model catalog (after login/logout)."""
    _status_cache["at"] = 0.0
    _status_cache["value"] = None
    _models_cache["at"] = 0.0
    _models_cache["value"] = None


def account_status(*, use_cache: bool = True) -> CodexStatus:
    """Evaluate the Codex login the same way the CLI's model loader does.

    Parameters
    ----------
    use_cache : bool, optional
        Reuse a lookup younger than :data:`STATUS_CACHE_TTL` seconds.

    Returns
    -------
    CodexStatus
        ``chatgpt`` is the only usable state; every other state carries a
        remedy in ``detail`` mirroring the CLI's hints.
    """
    cached = _status_cache["value"]
    if (
        use_cache
        and isinstance(cached, CodexStatus)
        and time.time() - float(_status_cache["at"]) < STATUS_CACHE_TTL
    ):
        return cached
    status = _account_status_uncached()
    _status_cache["at"] = time.time()
    _status_cache["value"] = status
    return status


def _account_status_uncached() -> CodexStatus:
    if not sdk_available():
        return CodexStatus(
            STATE_NO_SDK,
            "The openai-codex SDK is not installed. Run "
            "`pip install 'chemgraph[codex]'` in the UI's environment.",
        )
    from chemgraph.models.codex import account_summary, inspect_account

    try:
        summary = account_summary(inspect_account())
    except Exception as exc:  # SDK transport / app-server failures
        return CodexStatus(STATE_ERROR, f"Could not query the Codex login: {exc}")

    kind, identity = summary["type"], summary["identity"]
    if kind == "chatgpt":
        who = f" as {identity}" if identity else ""
        return CodexStatus(
            STATE_CHATGPT, f"Signed in to Codex with ChatGPT{who}.", identity
        )
    if kind == "apiKey":
        return CodexStatus(
            STATE_API_KEY,
            "The active Codex login uses an API key, which ChemGraph's codex: "
            "provider refuses. Log out, then sign in with ChatGPT.",
            identity,
        )
    return CodexStatus(
        STATE_LOGGED_OUT,
        "No Codex login is available. Sign in with ChatGPT to use codex: models.",
    )


def available_models(*, use_cache: bool = True) -> list[dict]:
    """Return the models the signed-in account can use, prefixed for ChemGraph.

    Each entry is ``{"name": "codex:<id>", "model", "display_name",
    "description", "is_default"}``; hidden models are omitted.  Empty when
    the login is not usable or the catalog cannot be fetched (the picker
    then falls back to a free-text model id).

    Parameters
    ----------
    use_cache : bool, optional
        Reuse a catalog younger than :data:`STATUS_CACHE_TTL` seconds.
    """
    cached = _models_cache["value"]
    if (
        use_cache
        and isinstance(cached, list)
        and time.time() - float(_models_cache["at"]) < STATUS_CACHE_TTL
    ):
        return cached
    models: list[dict] = []
    if account_status(use_cache=use_cache).ready:
        from chemgraph.models.codex import list_models

        try:
            for item in list_models():
                if item.get("hidden"):
                    continue
                models.append(
                    {
                        "name": f"{CODEX_MODEL_PREFIX}{item['model']}",
                        "model": item["model"],
                        "display_name": item.get("display_name") or item["model"],
                        "description": item.get("description", ""),
                        "is_default": bool(item.get("is_default")),
                    }
                )
        except Exception:  # SDK transport / app-server failures
            models = []
    _models_cache["at"] = time.time()
    _models_cache["value"] = models
    return models


def default_model_name(models: Optional[list[dict]] = None) -> Optional[str]:
    """Return the account's default ``codex:<id>`` model, if the catalog has one."""
    models = available_models() if models is None else models
    for item in models:
        if item.get("is_default"):
            return item["name"]
    return models[0]["name"] if models else None


def logout() -> tuple[bool, str]:
    """Sign out of the reusable Codex login through the SDK.

    Returns
    -------
    tuple[bool, str]
        ``(ok, message)``.
    """
    from chemgraph.models.codex import logout_account

    try:
        logout_account()
    except Exception as exc:  # SDK missing / transport failures
        return False, str(exc)
    finally:
        invalidate_status_cache()
    return True, "Signed out of Codex."


class DeviceLogin:
    """One device-code login attempt run by the SDK in a background thread.

    The thread owns the SDK session for the whole attempt, because the
    login handle is bound to it: it starts the login, publishes the
    verification URL and one-time code, then blocks on the SDK's
    completion notification.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._issued = threading.Event()
        self._done = threading.Event()
        self._handle: Any = None
        self.url: Optional[str] = None
        self.code: Optional[str] = None
        self.error: Optional[str] = None
        self.success: Optional[bool] = None
        self.cancelled = False
        self.started_at = time.time()
        self._thread: Optional[threading.Thread] = None

    # -- background thread -------------------------------------------------
    def _run(self) -> None:
        from chemgraph.models.codex import _model_dump, codex_session

        try:
            with codex_session() as codex:
                handle = codex.login_chatgpt_device_code()
                with self._lock:
                    self._handle = handle
                    self.url = str(handle.verification_url)
                    self.code = str(handle.user_code)
                self._issued.set()
                result = _model_dump(handle.wait())
            success = bool(result.get("success")) if isinstance(result, dict) else False
            error = result.get("error") if isinstance(result, dict) else None
            with self._lock:
                self.success = success and not self.cancelled
                if not success:
                    self.error = str(error or "Codex login did not complete.")
        except Exception as exc:  # SDK missing / app-server failures
            with self._lock:
                self.success = False
                self.error = str(exc) or type(exc).__name__
        finally:
            invalidate_status_cache()
            self._issued.set()
            self._done.set()

    def start(self, timeout: float = LOGIN_START_TIMEOUT) -> "DeviceLogin":
        """Start the attempt and wait until a code is issued (or it fails)."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._issued.wait(timeout)
        return self

    # -- state -------------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def succeeded(self) -> bool:
        return self.finished and self.success is True

    @property
    def output(self) -> str:
        """Human-readable failure detail (empty while pending or on success)."""
        return self.error or ""

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the attempt finishes; return whether it finished."""
        return self._done.wait(timeout)

    def cancel(self) -> None:
        """Cancel the attempt if it is still waiting for the code."""
        with self._lock:
            self.cancelled = True
            handle = self._handle
        if handle is not None and not self.finished:
            try:
                handle.cancel()
            except Exception:
                pass


def start_device_login() -> DeviceLogin:
    """Start an SDK device-code login and return a handle to poll.

    Raises
    ------
    RuntimeError
        When the SDK is missing or does not issue a sign-in code.
    """
    if not sdk_available():
        raise RuntimeError(
            "The openai-codex SDK is not installed. Run "
            "`pip install 'chemgraph[codex]'` in the UI's environment."
        )
    login = DeviceLogin().start()
    if login.code is None or login.url is None:
        if not login.finished:
            login.cancel()
        raise RuntimeError(
            "Codex did not issue a sign-in code"
            + (f": {login.error}" if login.error else ".")
        )
    return login


def is_codex_model(model_name: str) -> bool:
    """Return whether *model_name* routes to the Codex subscription adapter."""
    return bool(model_name) and model_name.startswith(CODEX_MODEL_PREFIX)


__all__ = [
    "CodexStatus",
    "DeviceLogin",
    "INSTALL_HINT",
    "READY_STATES",
    "LOGIN_START_TIMEOUT",
    "account_status",
    "available_models",
    "default_model_name",
    "invalidate_status_cache",
    "is_codex_model",
    "logout",
    "sdk_available",
    "start_device_login",
]
