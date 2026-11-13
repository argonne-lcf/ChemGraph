"""The artifact subcommand works offline and emits an unambiguous exit status."""

import importlib
import json
import subprocess
import sys

import pytest

from tests.test_artifact_audit import sample as sample

cli = importlib.import_module("chemgraph.cli.main")


@pytest.mark.parametrize("case,code", [("match", 0), ("broken", 1), ("root", 2), ("empty", 3)])
def test_cli_json_and_exit_codes(sample, monkeypatch, capsys, case, code):
    workspace, journal, record = sample
    if case == "broken":
        (workspace / record["result"]["artifacts"][0]["path"]).unlink()
    if case == "empty":
        record["result"]["artifacts"] = []
        record["prepared"]["inputs"] = []
    (journal / "op.json").write_text(json.dumps(record))
    if case == "root":
        journal = journal / "absent"
    def forbidden(*args, **kwargs):
        pytest.fail("Artifact inspection entered the agent/configuration path")
    for name in ("_handle_run", "initialize_agent", "check_api_keys_status"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(cli.toml, "load", forbidden)
    monkeypatch.setattr(sys, "argv", ["chemgraph", "artifacts", "verify", "--workspace", str(workspace),
                                     "--journal", str(journal), "--thread-id", "thread", "--json", "--include-inputs"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == code
    captured = capsys.readouterr()
    if code == 2:
        assert captured.out == "" and "Artifact audit" in captured.err
        assert not journal.exists()
    else:
        report = json.loads(captured.out)
        assert report["format_version"] == 1 and report["selection"]["include_inputs"]
        assert report["outcome"] == {0: "verified", 1: "issues", 3: "no_descriptors"}[code]


def test_verify_help_and_required_arguments(capsys):
    parser = cli.create_argument_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["artifacts", "verify", "--help"])
    assert exc.value.code == 0 and "Exit codes" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["artifacts", "verify"])
    assert exc.value.code == 2


def test_human_output_escapes_control_characters(sample, monkeypatch, capsys):
    workspace, journal, record = sample
    record["id"] = "[bold]\x1b[2J"
    (journal / "op.json").write_text(json.dumps(record))
    monkeypatch.setattr(sys, "argv", ["chemgraph", "artifacts", "verify", "--workspace", str(workspace),
                                     "--journal", str(journal), "--thread-id", "thread"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    text = capsys.readouterr().out
    assert exc.value.code == 0 and "[bold]\\x1b[2J" in text and "\x1b" not in text
    assert "Execution outcome: unknown" in text and "skipped 0" in text


def test_fresh_process_emits_only_json(sample):
    workspace, journal, record = sample
    (journal / "op.json").write_text(json.dumps(record))
    result = subprocess.run(
        [sys.executable, "-c", "from chemgraph.cli import main; main()", "artifacts", "verify",
         "--workspace", str(workspace), "--journal", str(journal), "--thread-id", "thread", "--json"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["outcome"] == "verified"
