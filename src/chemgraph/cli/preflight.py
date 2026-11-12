"""Advisory CLI readiness checks; remedies are always run by the user."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

from rich.console import Console
from rich.table import Table
from rich.text import Text

PROBE_TIMEOUT = 3.0
REPORT_TIMEOUT = 8.0
_GCP_INSTALL = "https://docs.globus.org/globus-connect-personal/install/"
_GLOBUS_INSTALL = 'uv pip install "chemgraph[globus]"'
_IRI_INSTALL = "uv pip install alcf-tokens"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    summary: str
    remedy: str = ""


def _unknown(name: str) -> CheckResult:
    command = {
        "Globus CLI authentication": "globus whoami",
        "Globus Connect Personal": "globus endpoint local-id, then globus endpoint show <id> --format json",
        "IRI authentication": "alcf-tokens test-token iri",
    }.get(name)
    remedy = (f"Inspect {command} in your terminal, then rerun doctor." if command else
              "Check connectivity, then rerun chemgraph doctor or /doctor.")
    return CheckResult(name, "Unverified", "Status unavailable; run the diagnostic command below.",
                       remedy)


def _run(argv: list[str], deadline: float) -> subprocess.CompletedProcess:
    """Bound each status command and clean up timed-out processes."""
    timeout = min(PROBE_TIMEOUT, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError
    process = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            # Terminate any children of the status command too.
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=1, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            process.kill()
        process.communicate()
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _gcp(globus: str, deadline: float) -> CheckResult:
    name = "Globus Connect Personal"
    local = _run([globus, "endpoint", "local-id"], deadline)
    if local.returncode:
        if "No Globus Connect Personal installation found." in local.stdout:
            return CheckResult(name, "Needs setup", "No local collection is configured.",
                               f"Install/configure Globus Connect Personal: {_GCP_INSTALL}")
        return _unknown(name)
    endpoint = str(uuid.UUID(local.stdout.strip()))
    result = _run([globus, "endpoint", "show", endpoint, "--format", "json"], deadline)
    if result.returncode == 4:
        return CheckResult(name, "Unverified", "Collection status requires Globus authorization.",
                           "Run globus login; rerun the endpoint show command for collection consent details.")
    if result.returncode:
        return _unknown(name)
    data = json.loads(result.stdout)
    if data.get("is_globus_connect") is not True:
        return _unknown(name)
    if data.get("gcp_paused") is True:
        return CheckResult(name, "Needs action", "Transfers are paused.",
                           "Resume transfers in the Globus Connect Personal application.")
    if data.get("gcp_connected") is True:
        return CheckResult(name, "Ready", "Local collection is connected; transfer permissions are not checked.")
    if data.get("gcp_connected") is False:
        launch = ("Run ./globusconnectpersonal -start from its installation directory."
                  if sys.platform.startswith("linux") else "Launch the Globus Connect Personal application.")
        return CheckResult(name, "Needs action", "Local collection is disconnected.",
                           f"{launch} If already running, check its network connection.")
    return _unknown(name)


def _globus(deadline: float) -> list[CheckResult]:
    name = "Globus CLI authentication"
    globus = shutil.which("globus")
    if not globus:
        return [CheckResult(name, "Missing", "Install ChemGraph's Globus support, then log in.",
                            f"{_GLOBUS_INSTALL}\nglobus login"),
                CheckResult("Globus Connect Personal", "Not checked", "Install Globus support first.")]
    try:
        result = _run([globus, "whoami"], deadline)
        if result.returncode == 0 and result.stdout.strip():
            auth = CheckResult(name, "Ready", "Login verified; collection-specific consent is not checked.")
        elif result.returncode == 4:
            auth = CheckResult(name, "Needs login", "Authentication or consent is required.", "globus login")
        else:
            auth = _unknown(name)
    except Exception:
        auth = _unknown(name)
    try:
        personal = _gcp(globus, deadline)
    except Exception:
        personal = _unknown("Globus Connect Personal")
    return [auth, personal]


def _iri(deadline: float) -> list[CheckResult]:
    name = "IRI authentication"
    executable = shutil.which("alcf-tokens")
    if not executable:
        return [CheckResult(name, "Missing", "alcf-tokens is not on PATH.",
                            f"{_IRI_INSTALL}\nalcf-tokens login iri")]
    result = _run([executable, "test-token", "iri"], deadline)
    data = json.loads(result.stdout)
    if result.returncode == 0 and data.get("ready") is True:
        return [CheckResult(name, "Ready", "IRI skill credentials verified by alcf-tokens.")]
    error = data.get("error")
    if data.get("ready") is False and isinstance(error, str) and (
        error.startswith("No valid tokens found for ") or re.match(r"HTTP (401|403)\b", error)
    ):
        return [CheckResult(name, "Needs login", "IRI authentication or consent is required.",
                            "alcf-tokens login iri")]
    return [_unknown(name)]


def collect_checks() -> list[CheckResult]:
    """Collect independent probes without prompting, making model calls, or exposing errors."""
    deadline = time.monotonic() + REPORT_TIMEOUT
    probes = [
        (("Globus CLI authentication", "Globus Connect Personal"), lambda: _globus(deadline)),
        (("IRI authentication",), lambda: _iri(deadline)),
    ]

    def safe_probe(names, probe):
        try:
            return probe()
        except Exception:
            return [_unknown(name) for name in names]

    with ThreadPoolExecutor(max_workers=len(probes)) as pool:
        futures = [pool.submit(safe_probe, names, probe) for names, probe in probes]
        return [result for future in futures for result in future.result()]


def show_report(*, console: Console | None = None) -> None:
    """Render only deliberately constructed summaries, never raw subprocess output."""
    console = console if console is not None else Console(stderr=True)
    table = Table(title="ChemGraph readiness (advisory)")
    for column in ("Prerequisite", "Status", "Details"):
        table.add_column(column)
    try:
        results = collect_checks()
    except Exception:
        results = [_unknown("Readiness checks")]
    for result in results:
        style = {"Ready": "green", "Not checked": "dim"}.get(result.status, "yellow")
        table.add_row(Text(result.name), Text(result.status, style=style), Text(result.summary))
    console.print(table)
    remedies = list(dict.fromkeys(result.remedy for result in results if result.remedy))
    if remedies:
        console.print("\nRun these setup or diagnostic steps in another terminal:")
        for remedy in remedies:
            console.print(Text(remedy), soft_wrap=True)
    console.print("\nGlobus and IRI are optional for remote work. Recheck with chemgraph doctor or /doctor.")
