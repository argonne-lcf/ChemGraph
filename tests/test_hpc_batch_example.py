"""The live example uses real run evidence and its shell job, with offline services."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock

import pytest

from chemgraph.tools.hpc.tools import create_hpc_registry
from tests.test_hpc_batch import batch as batch


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/hpc_batch"
spec = importlib.util.spec_from_file_location("hpc_batch_demo", EXAMPLE / "run_demo.py")
demo = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
try:
    spec.loader.exec_module(demo)
finally:
    sys.path[:] = original_path


@pytest.mark.parametrize("initializer", [None, "", "raise RuntimeError('private-detail')\n"])
def test_incomplete_checkout_stops_cleanly_without_loading_installed_copy(tmp_path, initializer):
    example = tmp_path / "checkout/examples/hpc_batch"
    example.mkdir(parents=True)
    shutil.copyfile(EXAMPLE / "run_demo.py", example / "run_demo.py")
    package = tmp_path / "checkout/src/chemgraph"
    package.mkdir(parents=True)
    if initializer is not None:
        (package / "__init__.py").write_text(initializer)
    installed = tmp_path / "installed/chemgraph"
    installed.mkdir(parents=True)
    touched = tmp_path / "wrong-checkout-imported"
    (installed / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(touched)!r}).touch()\n"
    )
    root = tmp_path / "existing-run"
    root.mkdir()
    (root / "submission.started").write_text("existing-attempt")
    result = subprocess.run(
        [sys.executable, str(example / "run_demo.py"), "collect", "--config", "unused.toml",
         "--run-dir", str(root)], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(installed.parent)}, capture_output=True, text=True,
    )
    assert result.returncode == 1
    stopped = json.loads(result.stdout)
    assert stopped["step"] == "stopped"
    if initializer is None:
        assert "checkout is incomplete" in stopped["message"]
    elif not initializer:
        assert stopped["type"] == "ModuleNotFoundError"
        assert "Cannot load the demo runtime" in stopped["message"]
    else:
        assert stopped["type"] == "RuntimeError"
    assert "Traceback" not in result.stderr
    assert "private-detail" not in result.stdout + result.stderr
    assert not touched.exists()
    assert list(root.iterdir()) == [root / "submission.started"]
    assert (root / "submission.started").read_text() == "existing-attempt"


def test_already_imported_other_checkout_stops_before_hpc_imports(tmp_path):
    package = tmp_path / "other/chemgraph"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    touched = tmp_path / "child-imported"
    for child in ("tools", "execution"):
        (package / child).mkdir()
        (package / child / "__init__.py").write_text(
            f"from pathlib import Path\nPath({str(touched)!r}).touch()\n"
        )
    bootstrap = (
        "import chemgraph, runpy, sys\n"
        f"demo = runpy.run_path({str(EXAMPLE / 'run_demo.py')!r})\n"
        "sys.exit(demo['main'](['check', '--config', 'unused.toml']))\n"
    )
    result = subprocess.run([sys.executable, "-c", bootstrap], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(package.parent)},
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "Another ChemGraph checkout is loaded" in json.loads(result.stdout)["message"]
    assert "Traceback" not in result.stderr
    assert not touched.exists()


def test_missing_config_error_does_not_import_sdk_in_exception_handler(tmp_path):
    bootstrap = f"""
import builtins, runpy, sys
demo = runpy.run_path({str(EXAMPLE / 'run_demo.py')!r})
load_config = demo['load_config']
original_import = builtins.__import__
def unavailable_sdk(name, *args, **kwargs):
    if name == 'globus_sdk':
        raise ImportError('private-sdk-error')
    return original_import(name, *args, **kwargs)
def missing_config(*args):
    assert 'globus_sdk' in sys.modules
    builtins.__import__ = unavailable_sdk
    return load_config(*args)
demo['main'].__globals__['load_config'] = missing_config
sys.exit(demo['main'](['check', '--config', 'missing.toml']))
"""
    result = subprocess.run([sys.executable, "-c", bootstrap], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 1
    stopped = json.loads(result.stdout)
    assert stopped["step"] == "stopped" and stopped["type"] == "FileNotFoundError"
    assert "Traceback" not in result.stderr
    assert "private-sdk-error" not in result.stdout + result.stderr


def test_placeholder_config_stops_before_any_client_is_created(monkeypatch, capsys):
    factory = Mock(side_effect=AssertionError("must not contact a service"))
    monkeypatch.setattr("chemgraph.tools.hpc.service.HPCService", factory)
    assert demo.main(["check", "--config", str(EXAMPLE / "config.example.toml")]) == 1
    assert "Replace every YOUR_* placeholder" in capsys.readouterr().out
    factory.assert_not_called()


def test_disconnected_collection_has_actionable_redacted_diagnostic(batch):
    from tests.test_hpc_batch import api_error

    service, _, transfer, *_ = batch
    client = Mock()
    client.operation_ls.side_effect = api_error(502, "ExternalError.DirListingFailed.GCDisconnected")
    transfer._get_transfer_client = Mock(return_value=client)
    with pytest.raises(demo.DemoError, match="Start Globus Connect Personal") as exc:
        demo.preflight(service, "polaris")
    assert '"side": "local"' in str(exc.value)
    assert "private-" not in str(exc.value)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_wait_limits_are_rejected_before_accessing_services(value):
    with pytest.raises(SystemExit) as exc:
        demo.main(["check", "--config", str(EXAMPLE / "config.example.toml"), "--timeout", value])
    assert exc.value.code == 2


def offline_demo(batch, monkeypatch, *, corrupt=False):
    service, old_root, transfer, iri, _ = batch
    submit = iri.submit_prepared

    def execute(prepared):
        accepted = submit(prepared)
        _, job = prepared
        with open(job["stdout_path"], "w") as stdout, open(job["stderr_path"], "w") as stderr:
            # PBS can start outside the staged directory despite the jobspec directory.
            subprocess.run([job["executable"], *job["arguments"]], cwd=old_root.parent,
                           env={**os.environ, "PBS_JOBID": accepted["id"]},
                           stdout=stdout, stderr=stderr, check=True)
        if corrupt:
            result = Path(job["directory"]) / "result.txt"
            result.write_text(result.read_text().replace("sum=55", "sum=54"))
        iri.finished = True
        return accepted

    monkeypatch.setattr(iri, "submit_prepared", execute)
    registry = create_hpc_registry(service.config, service=service)
    return service, old_root.parent / "demo", transfer, iri, registry


@pytest.mark.parametrize("uppercase", [False, True])
def test_demo_runs_shell_job_verifies_outputs_and_collects_without_duplicates(batch, monkeypatch, capsys, uppercase):
    service, root, transfer, iri, registry = offline_demo(batch, monkeypatch)
    if uppercase:
        monkeypatch.setattr(iri, "status", lambda *args, **kwargs: {"status": {"state": "COMPLETED", "exit_code": 0}})
    demo.run(service.config, registry, "polaris", root, 1, 0)
    assert '"step": "verified"' in capsys.readouterr().out
    assert "sum=55" in (root / "retrieved/result.txt").read_text()
    evidence = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    demo.collect(registry, root, 1, 0)
    with pytest.raises(FileExistsError):
        demo.run(service.config, registry, "polaris", root, 1, 0)
    assert len(iri.submissions) == 1 and len(transfer.calls) == 2
    for name in ("run.json", "submission.json", "submission.started", ".hpc-retrievals.json"):
        assert (root / name).read_bytes() == evidence[name]


def test_demo_does_not_treat_scheduler_completion_as_verified_output(batch, monkeypatch):
    service, root, transfer, iri, registry = offline_demo(batch, monkeypatch, corrupt=True)
    with pytest.raises(demo.DemoError, match="output_checks_failed") as exc:
        demo.run(service.config, registry, "polaris", root, 1, 0)
    assert json.loads(str(exc.value))["output_checks_failed"] == {"sum": {"expected": "55", "observed": "54"}}
    assert len(iri.submissions) == 1 and len(transfer.calls) == 2
    assert (root / "submission.started").exists()


@pytest.mark.parametrize("state", ["ACTIVE", "FAILED"])
def test_demo_staging_failure_or_timeout_never_submits_computation(batch, state):
    service, old_root, transfer, iri, _ = batch
    transfer.state = state
    registry = create_hpc_registry(service.config, service=service)
    root = old_root.parent / "demo"
    with pytest.raises(demo.DemoError):
        demo.run(service.config, registry, "polaris", root, 0, 0)
    assert (root / "run.json").exists()
    assert not iri.submissions and len(transfer.calls) == 1


def test_demo_keeps_unknown_submission_evidence_without_retrying(batch):
    service, old_root, transfer, iri, _ = batch
    iri.failure = TimeoutError("response lost")
    registry = create_hpc_registry(service.config, service=service)
    root = old_root.parent / "demo"
    with pytest.raises(demo.DemoError):
        demo.run(service.config, registry, "polaris", root, 1, 0)
    assert (root / "submission.started").exists()
    assert len(iri.submissions) == 1 and len(transfer.calls) == 1
    evidence = json.loads((root / "submission.json").read_text())
    assert len(evidence["attempts"]) == 1
    assert evidence["attempts"][0]["state"] == "submission_unknown"
    assert evidence["attempts"][0]["error"] == {"type": "TimeoutError"}
    with pytest.raises(FileExistsError):
        demo.run(service.config, registry, "polaris", root, 1, 0)
    assert len(iri.submissions) == 1


@pytest.mark.parametrize("state", ["ACTIVE", "FAILED"])
def test_demo_waits_on_recorded_retrieval_after_timeout(batch, monkeypatch, capsys, state):
    service, root, transfer, iri, registry = offline_demo(batch, monkeypatch)
    submit = transfer.submit_prepared

    def slow_retrieval(prepared):
        task = submit(prepared)
        if len(transfer.calls) == 2:
            transfer.state = state
        return task

    monkeypatch.setattr(transfer, "submit_prepared", slow_retrieval)
    with pytest.raises(demo.DemoError, match="Timed out" if state == "ACTIVE" else "failed"):
        demo.run(service.config, registry, "polaris", root, 0, 0)
    capsys.readouterr()
    demo.show_status(registry, root)
    status = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert status[-1]["step"] == "retrieval status" and status[-1]["status"] == state
    if state == "ACTIVE":
        transfer.state = "SUCCEEDED"
        demo.collect(registry, root, 1, 0)
    else:
        with pytest.raises(demo.DemoError, match="failed"):
            demo.collect(registry, root, 1, 0)
    assert len(transfer.calls) == 2 and len(iri.submissions) == 1


def test_demo_reports_malformed_retrieved_result(batch, monkeypatch):
    service, root, transfer, iri, registry = offline_demo(batch, monkeypatch)
    demo.run(service.config, registry, "polaris", root, 1, 0)
    (root / "retrieved/result.txt").write_text("not a key-value result\n")
    with pytest.raises(demo.DemoError, match="Missing or malformed"):
        demo.collect(registry, root, 1, 0)
    assert len(transfer.calls) == 2 and len(iri.submissions) == 1
