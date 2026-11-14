"""Offline readiness probes and CLI integration."""

import importlib
import io
import json
import subprocess
import sys
import time

import pytest
from rich.console import Console

from chemgraph.cli import commands, preflight

cli_main = importlib.import_module("chemgraph.cli.main")
run_process = preflight._run
ENDPOINT = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def probes(monkeypatch):
    responses = {
        "whoami": (0, "person@example.org"),
        "local-id": (0, ENDPOINT),
        "show": (0, {"is_globus_connect": True, "gcp_connected": True, "gcp_paused": False}),
        "test-token": (0, {"ready": True}),
    }
    calls = []

    def run(argv, deadline):
        calls.append(argv)
        assert deadline > time.monotonic()
        key = argv[2] if argv[1] == "endpoint" else argv[1]
        result = responses[key]
        if isinstance(result, Exception):
            raise result
        code, data = result
        return subprocess.CompletedProcess(argv, code, json.dumps(data) if isinstance(data, dict) else data,
                                           "secret-token [bold]must never appear[/bold]")

    monkeypatch.setattr(preflight.shutil, "which", lambda name: name)
    monkeypatch.setattr(preflight, "_run", run)
    return responses, calls


def results():
    return {result.name: result for result in preflight.collect_checks()}


def test_ready_report_uses_stderr_without_exposing_credentials(probes, capsys):
    report = results()
    assert all(row.status == "Ready" for row in report.values())
    assert set(report) == {"Globus CLI authentication", "Globus Connect Personal", "IRI authentication"}
    preflight.show_report()
    captured = capsys.readouterr()
    text = captured.err
    assert "ChemGraph readiness" in text and not captured.out
    assert "secret-token" not in text and "person@example.org" not in text
    assert "optional" in text and "/doctor" in text
    assert all("login" not in call and "get-token" not in call for call in probes[1])


def test_missing_tools_do_not_claim_gcp_is_uninstalled(probes, monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    report = results()
    assert report["Globus CLI authentication"].status == "Missing"
    assert report["Globus Connect Personal"].status == "Not checked"
    assert "alcf-tokens login iri" in report["IRI authentication"].remedy
    output = io.StringIO()
    preflight.show_report(console=Console(file=output, width=60))
    assert output.getvalue().count('uv pip install "chemgraph[globus]"') == 1
    assert not probes[1]


@pytest.mark.parametrize("response,status", [
    ((1, {"ready": False, "error": "No valid tokens found for 'iri' (secret-token)."}), "Needs login"),
    ((1, {"ready": False, "error": "HTTP 401: secret-token"}), "Needs login"),
    ((1, {"ready": False, "error": "HTTP 503: unavailable"}), "Unverified"),
    ((1, {"ready": True}), "Unverified"),
    ((0, "unsupported command output"), "Unverified"),
    (TimeoutError("secret-token"), "Unverified"),
])
def test_iri_distinguishes_auth_from_unknown(probes, response, status):
    probes[0]["test-token"] = response
    row = results()["IRI authentication"]
    assert row.status == status
    assert "secret-token" not in repr(row)
    assert ("login iri" in row.remedy) == (status == "Needs login")


@pytest.mark.parametrize("code,status", [(4, "Needs login"), (1, "Unverified")])
def test_globus_failure_classification(probes, code, status):
    probes[0]["whoami"] = (code, "secret-token")
    assert results()["Globus CLI authentication"].status == status


@pytest.mark.parametrize("response,status,summary", [
    ((0, {"is_globus_connect": True, "gcp_connected": False}), "Needs action", "disconnected"),
    ((0, {"is_globus_connect": True, "gcp_connected": True, "gcp_paused": True}), "Needs action", "paused"),
    ((0, {"is_globus_connect": True}), "Unverified", "unavailable"),
    ((4, "secret-token"), "Unverified", "authorization"),
])
def test_gcp_states(probes, response, status, summary):
    probes[0]["show"] = response
    report = results()
    assert report["Globus CLI authentication"].status == "Ready"
    assert report["Globus Connect Personal"].status == status
    assert summary in report["Globus Connect Personal"].summary


def test_gcp_absent_and_invalid_id_do_not_query_remote(probes):
    probes[0]["local-id"] = (1, "No Globus Connect Personal installation found.")
    assert results()["Globus Connect Personal"].status == "Needs setup"
    output = io.StringIO()
    preflight.show_report(console=Console(file=output, width=60))
    assert "https://docs.globus.org/globus-connect-personal/install/" in output.getvalue()
    probes[0]["local-id"] = (0, "invalid; secret-token")
    assert results()["Globus Connect Personal"].status == "Unverified"
    assert not any("show" in call for call in probes[1])


def test_probe_runner_closed_stdin_and_cleanup(monkeypatch):
    created = []
    popen = subprocess.Popen

    def record(argv, **kwargs):
        if argv[0] == sys.executable:
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert not kwargs.get("shell")
        process = popen(argv, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(preflight.subprocess, "Popen", record)
    monkeypatch.setattr(preflight, "PROBE_TIMEOUT", 0.2)
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], time.monotonic() + 1)
    assert created and all(process.poll() is not None for process in created)
    count = len(created)
    with pytest.raises(TimeoutError):
        run_process(["probe"], time.monotonic() - 1)
    assert len(created) == count


@pytest.mark.parametrize("workflow,enabled,reported", [
    ("main_agent", True, True), ("deepagent", True, True),
    ("single_agent", True, False), ("main_agent", False, False),
])
def test_startup_report_precedes_backend_and_model_validation(monkeypatch, workflow, enabled, reported):
    events = []
    monkeypatch.setattr(commands, "show_report", lambda *args, **kwargs: events.append("report"))
    monkeypatch.setattr(commands, "_create_experimental_deepagent_backend",
                        lambda *args, **kwargs: events.append("backend"))
    monkeypatch.setattr(commands, "check_api_keys", lambda *args, **kwargs: (events.append("keys") or False, "missing"))
    assert commands.initialize_agent("test", workflow, False, "state", False, 10, preflight=enabled) is None
    assert ("report" in events) == reported
    if "report" in events:
        assert events[0] == "report"
    assert events[-1] == "keys"


@pytest.mark.parametrize("prefix,workflow,interactive,enabled", [
    (["--no-preflight", "run"], "main_agent", True, False),
    (["run", "--no-preflight"], "main_agent", True, False),
    (["--no-preflight", "run"], "main_agent", False, False),
    (["run", "--no-preflight"], "main_agent", False, False),
    (["--no-preflight"], "deepagent", False, False),
    (["run"], "deepagent", False, True),
    ([], "main_agent", False, True),
])
def test_preflight_dispatch(monkeypatch, tmp_path, prefix, workflow, interactive, enabled):
    captured = []
    monkeypatch.setattr(cli_main, "interactive_mode", lambda **kwargs: captured.append(kwargs))
    monkeypatch.setattr(cli_main, "run_headless_main_agent", lambda **kwargs: captured.append(kwargs))
    monkeypatch.setattr(cli_main, "initialize_agent", lambda *args, **kwargs: captured.append(kwargs))
    argv = [*prefix, "-w", workflow, "--dangerously-skip-approvals", "-q", "test"]
    argv += ["--interactive"] if interactive else ["--deepagent-workspace", str(tmp_path)] if workflow == "deepagent" else []
    try:
        cli_main._handle_run(cli_main.create_argument_parser().parse_args(argv))
    except SystemExit as exc:
        assert exc.code == 1  # mocked standalone initialization returns None
    assert captured and captured[0]["preflight"] is enabled


def test_doctor_without_agent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chemgraph", "doctor"])
    seen = []
    monkeypatch.setattr(preflight, "show_report", lambda *args, **kwargs: seen.append((args, kwargs)))
    monkeypatch.setattr(cli_main, "initialize_agent", lambda *args, **kwargs: pytest.fail("doctor constructed agent"))
    assert cli_main.main() is None
    assert seen == [((), {"console": cli_main.console})]
