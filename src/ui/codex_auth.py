"""Codex (ChatGPT subscription) login status and device-code login for the UI.

The CLI authorizes ``codex:<model-id>`` models through the login that the
Codex CLI stores (``codex login``); ChemGraph then validates that the
active login is ChatGPT-managed rather than an API key.  This module
exposes the same checks to the Streamlit UI and adds a way to start the
CLI's headless device-code login (``codex login --device-auth``) without
leaving the browser.  The subprocess prints a URL and a one-time code,
which the page shows to the user; the process exits on its own once the
code is redeemed.

Streamlit-free so it can be unit-tested with a mocked SDK/subprocess.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from chemgraph.models.codex import CODEX_MODEL_PREFIX

#: Login states reported by :func:`account_status`.
STATE_NO_SDK = "no_sdk"
STATE_NO_CLI = "no_cli"
STATE_LOGGED_OUT = "logged_out"
STATE_API_KEY = "api_key"
STATE_CHATGPT = "chatgpt"
STATE_ERROR = "error"

READY_STATES = frozenset({STATE_CHATGPT})

INSTALL_HINT = (
    "Install the Codex CLI separately, install the optional "
    "`chemgraph[codex]` extra, then sign in with ChatGPT authentication."
)

_URL_RE = re.compile(r"https?://\S+")
_CODE_RE = re.compile(r"\b[A-Z0-9]{4,6}-[A-Z0-9]{4,6}\b")
_SUCCESS_RE = re.compile(r"successfully logged in", re.IGNORECASE)


@dataclass(frozen=True)
class CodexStatus:
    """Readiness of the Codex subscription route."""

    state: str
    detail: str
    identity: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.state in READY_STATES


def codex_cli_path() -> Optional[str]:
    """Return the ``codex`` executable on ``PATH``, if any."""
    return shutil.which("codex")


def sdk_available() -> bool:
    """Return whether the optional ``openai-codex`` SDK is importable."""
    try:
        import openai_codex  # noqa: F401
    except ImportError:
        return False
    return True


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
    if codex_cli_path() is None:
        return CodexStatus(
            STATE_NO_CLI,
            "The `codex` CLI was not found on PATH. Install it from "
            "developers.openai.com/codex/cli, then sign in with ChatGPT.",
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
    """Run ``codex logout`` and return ``(ok, output)``."""
    cli = codex_cli_path()
    if cli is None:
        return False, "The `codex` CLI was not found on PATH."
    try:
        proc = subprocess.run(
            [cli, "logout"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


@dataclass
class DeviceLogin:
    """A running ``codex login --device-auth`` process and its parsed output."""

    process: subprocess.Popen
    started_at: float = field(default_factory=time.time)
    _lines: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader: Optional[threading.Thread] = None

    def start_reader(self) -> None:
        """Drain the subprocess output in a background thread."""

        def _pump():
            stream = self.process.stdout
            if stream is None:
                return
            for raw in iter(stream.readline, ""):
                with self._lock:
                    self._lines.append(raw.rstrip("\r\n"))

        self._reader = threading.Thread(target=_pump, daemon=True)
        self._reader.start()

    def feed(self, text: str) -> None:
        """Append output text (used by tests and non-threaded readers)."""
        with self._lock:
            self._lines.extend(text.splitlines())

    @property
    def output(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    @property
    def url(self) -> Optional[str]:
        for line in self.output.splitlines():
            match = _URL_RE.search(line)
            if match:
                return match.group(0).rstrip(".,)")
        return None

    @property
    def code(self) -> Optional[str]:
        # The code is printed on its own line after the instructions; the
        # first token that looks like XXXXX-XXXXX is the one-time code.
        for line in self.output.splitlines():
            match = _CODE_RE.search(line)
            if match and "://" not in line:
                return match.group(0)
        return None

    @property
    def finished(self) -> bool:
        return self.process.poll() is not None

    @property
    def succeeded(self) -> bool:
        return self.finished and (
            self.process.returncode == 0 or bool(_SUCCESS_RE.search(self.output))
        )

    def cancel(self) -> None:
        """Terminate the login process if it is still waiting."""
        if not self.finished:
            try:
                self.process.terminate()
            except OSError:
                pass


def start_device_login() -> DeviceLogin:
    """Start ``codex login --device-auth`` and return a handle to poll.

    Raises
    ------
    RuntimeError
        When the CLI is missing or cannot be started.
    """
    cli = codex_cli_path()
    if cli is None:
        raise RuntimeError(
            "The `codex` CLI was not found on PATH; install it before logging in."
        )
    try:
        process = subprocess.Popen(
            [cli, "login", "--device-auth"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start `codex login --device-auth`: {exc}") from exc
    login = DeviceLogin(process=process)
    login.start_reader()
    return login


def is_codex_model(model_name: str) -> bool:
    """Return whether *model_name* routes to the Codex subscription adapter."""
    return bool(model_name) and model_name.startswith(CODEX_MODEL_PREFIX)


__all__ = [
    "CodexStatus",
    "DeviceLogin",
    "INSTALL_HINT",
    "READY_STATES",
    "account_status",
    "available_models",
    "codex_cli_path",
    "default_model_name",
    "invalidate_status_cache",
    "is_codex_model",
    "logout",
    "sdk_available",
    "start_device_login",
]
