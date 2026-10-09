"""Execute packaged application and allocation assets without live services."""

import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

import pytest


SKILLS = Path(__file__).resolve().parents[1] / "src/chemgraph/skills"
requires_pbs_shell = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="PBS launch assets require a POSIX host with bash.",
)


@pytest.fixture
def launch_environment(tmp_path):
    environment = {"PATH": os.environ.get("PATH", os.defpath)}
    for name in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update(
        PYTHONPATH=str(SKILLS.parents[1]),
        PYTHONNOUSERSITE="1",
        CHEMGRAPH_LOG_DIR=str(tmp_path / "inherited-logs"),
    )
    return environment


@pytest.mark.parametrize("outcome", ["success", "nonconverged", "missing", "invalid", "no_input"])
def test_ase_runner_without_scheduler(tmp_path, launch_environment, outcome):
    shutil.copyfile(SKILLS / "chemgraph/assets/calculate.py", tmp_path / "calculate.py")
    (tmp_path / "water.xyz").write_text("3\nwater\nO 0 0 0\nH 0 0 1\nH 1 0 0\n")
    params = {
        "input_structure_file": "missing.xyz" if outcome == "missing" else "water.xyz",
        "output_results_file": "result.json",
        "driver": "invalid" if outcome == "invalid" else "opt",
        "steps": 0 if outcome == "nonconverged" else 100,
        "calculator": {"calculator_type": "emt"},
    }
    if outcome != "no_input":
        (tmp_path / "input.json").write_text(json.dumps(params))
    result = subprocess.run(
        [sys.executable, "calculate.py"], cwd=tmp_path, env=launch_environment,
        capture_output=True, text=True,
    )
    expected = {"success": 0, "nonconverged": 2}.get(outcome, 1)
    assert result.returncode == expected, result.stderr
    provenance = json.loads(result.stdout.splitlines()[0])
    assert provenance == {
        "compute_hostname": socket.gethostname(),
        "python": sys.executable,
        "cwd": str(tmp_path.resolve()),
    }
    assert not (tmp_path / "inherited-logs").exists()
    if outcome in {"success", "nonconverged"}:
        output = json.loads((tmp_path / "result.json").read_text())
        assert output["success"]
        assert output["converged"] == (outcome == "success")


@requires_pbs_shell
@pytest.mark.parametrize(
    "allocation",
    ["valid", "qualified", "no_job", "no_nodefile", "absent", "directory", "unreadable", "empty", "wrong_host"],
)
def test_pbs_helper_checks_allocation(tmp_path, launch_environment, allocation):
    helper = SKILLS / "pbs-hpc/assets/pbs-launch.sh"
    host = socket.gethostname().split(".")[0]
    nodefile = tmp_path / "nodes"
    nodefile.write_text(
        f"other.example\n\t{host}.example  {host}\n\n{host}.example"
        if allocation == "qualified" else host
    )
    launch_environment.update(PBS_JOBID="123.server", PBS_NODEFILE=str(nodefile))
    if allocation == "no_job":
        launch_environment.pop("PBS_JOBID")
    elif allocation == "no_nodefile":
        launch_environment.pop("PBS_NODEFILE")
    elif allocation == "absent":
        launch_environment["PBS_NODEFILE"] = str(tmp_path / "absent")
    elif allocation == "directory":
        launch_environment["PBS_NODEFILE"] = str(tmp_path)
    elif allocation == "unreadable":
        if os.geteuid() == 0:
            pytest.skip("Root can read mode-000 files.")
        nodefile.chmod(0)
    elif allocation == "empty":
        nodefile.write_text("")
    elif allocation == "wrong_host":
        nodefile.write_text(f"{host}-different.example")
    result = subprocess.run(
        ["bash", str(helper), "bash", "-c", "printf launched"],
        cwd=tmp_path, env=launch_environment, capture_output=True, text=True,
    )
    valid = allocation in {"valid", "qualified"}
    assert result.returncode == (0 if valid else 1), result.stderr
    assert result.stdout == ("launched" if valid else "")
    if valid:
        assert f"job=123.server host={host}" in result.stderr


@requires_pbs_shell
@pytest.mark.parametrize("skill", ["pbs-hpc", "iri-hpc"])
@pytest.mark.parametrize("allocated", [True, False])
def test_launch_templates_forward_arguments_and_status(tmp_path, launch_environment, skill, allocated):
    run = tmp_path / "run with spaces"
    run.mkdir()
    helper = run / "pbs-launch.sh"
    shutil.copyfile(SKILLS / "pbs-hpc/assets/pbs-launch.sh", helper)
    helper.chmod(0o400)  # Matches the immutable staging snapshot's permissions.
    application = run / "application script.sh"
    application.write_text('printf "%s\\n" "$LAUNCH_TEST" "$@"\nexit 7\n')
    setup = run / "environment setup.sh"
    setup.write_text('export LAUNCH_TEST="${UNSET_TEST_VARIABLE:-ready}"\n')
    template = "job.pbs.template" if skill == "pbs-hpc" else "launch.sh.template"
    script = (SKILLS / skill / "assets" / template).read_text()
    values = {
        "ENVIRONMENT_SETUP": f"source {shlex.quote(str(setup))}",
        "APPLICATION_LAUNCH_COMMAND": f"bash {shlex.quote(str(application))} 'fixed argument'",
        "JOB_NAME": "test", "PROJECT": "project", "QUEUE": "debug",
        "NODES": "1", "SYSTEM": "polaris", "WALLTIME": "00:05:00",
        "FILESYSTEMS": "home:eagle",
    }
    for key, value in values.items():
        script = script.replace("{{" + key + "}}", value)
    launch = run / "launch.sh"
    launch.write_text(script)
    assert "{{" not in script
    subprocess.run(["bash", "-n", str(launch)], check=True)
    nodefile = run / "nodes"
    nodefile.write_text(socket.gethostname())
    launch_environment.update(PBS_NODEFILE=str(nodefile), PBS_O_WORKDIR=str(run))
    if allocated:
        launch_environment["PBS_JOBID"] = "123.server"
    arguments = ["argument with spaces", "", "$(touch unexpected)", "*.xyz"]
    result = subprocess.run(
        ["bash", str(launch), *arguments],
        cwd=tmp_path if skill == "pbs-hpc" else run,
        env=launch_environment, capture_output=True, text=True,
    )
    assert result.returncode == (7 if allocated else 1), result.stderr
    assert result.stdout.splitlines() == (["ready", "fixed argument", *arguments] if allocated else [])
    assert not (run / "unexpected").exists()
