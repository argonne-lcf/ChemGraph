"""Offline readiness probes and CLI integration."""

import importlib
import json
import subprocess
import sys
import time

import pytest

from chemgraph.cli import preflight

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

    def run(argv, deadline):
        key = argv[2] if argv[1] == "endpoint" else argv[1]
        result = responses[key]
        if isinstance(result, Exception):
            raise result
        code, data = result
        return subprocess.CompletedProcess(argv, code, json.dumps(data) if isinstance(data, dict) else data,
                                           "secret-token [bold]must never appear[/bold]")

    monkeypatch.setattr(preflight.shutil, "which", lambda name: name)
    monkeypatch.setattr(preflight, "_run", run)
    return responses


def results():
    return {result.name: result for result in preflight.collect_checks()}


def test_ready_report_uses_stderr_without_exposing_credentials(probes, capsys):
    report = results()
    assert [row.status for row in report.values()] == ["Ready"] * 3
    preflight.show_report()
    captured = capsys.readouterr()
    text = captured.err
    assert "ChemGraph readiness" in text and not captured.out
    assert "secret-token" not in text and "person@example.org" not in text


def test_missing_tools_do_not_claim_gcp_is_uninstalled(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    assert [row.status for row in preflight.collect_checks()] == ["Missing", "Not checked", "Missing"]


@pytest.mark.parametrize("command,name,response,status", [
    ("whoami", "Globus CLI authentication", (4, "secret-token"), "Needs login"),
    ("test-token", "IRI authentication", (1, {"ready": False, "error": "HTTP 401: secret-token"}), "Needs login"),
    ("test-token", "IRI authentication", (1, {"ready": False, "error": "HTTP 503: unavailable"}), "Unverified"),
    ("test-token", "IRI authentication", TimeoutError("secret-token"), "Unverified"),
    ("show", "Globus Connect Personal", (0, {"is_globus_connect": True, "gcp_connected": False}), "Needs action"),
    ("show", "Globus Connect Personal", (0, {"is_globus_connect": True, "gcp_paused": True}), "Needs action"),
])
def test_probe_statuses(probes, command, name, response, status):
    probes[command] = response
    row = results()[name]
    assert row.status == status
    assert "secret-token" not in repr(row)
    assert ("login" in row.remedy) == (status == "Needs login")


def test_probe_runner_closed_stdin_and_cleanup(monkeypatch):
    created = []
    popen = subprocess.Popen

    def record(argv, **kwargs):
        if argv[0] == sys.executable:
            assert kwargs["stdin"] == subprocess.DEVNULL
        process = popen(argv, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(preflight.subprocess, "Popen", record)
    monkeypatch.setattr(preflight, "PROBE_TIMEOUT", 0.2)
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], time.monotonic() + 1)
    assert created and all(process.poll() is not None for process in created)


@pytest.mark.parametrize("prefix,workflow,interactive", [
    (["--no-preflight", "run"], "main_agent", False),
    (["run", "--no-preflight"], "main_agent", True),
    (["--no-preflight"], "deepagent", False),
])
def test_preflight_dispatch(monkeypatch, tmp_path, prefix, workflow, interactive):
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
    assert captured and captured[0]["preflight"] is False


def test_doctor_without_agent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chemgraph", "doctor"])
    seen = []
    monkeypatch.setattr(preflight, "show_report", lambda *args, **kwargs: seen.append((args, kwargs)))
    monkeypatch.setattr(cli_main, "initialize_agent", lambda *args, **kwargs: pytest.fail("doctor constructed agent"))
    assert cli_main.main() is None
    assert seen == [((), {"console": cli_main.console})]
