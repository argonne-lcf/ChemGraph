"""Packaged PBS example staging and submission approvals, without a scheduler."""

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


@pytest.mark.skipif(os.name == "nt", reason="PBS submission requires a POSIX shell")
def test_rendered_batch_template_runs_staged_entrypoint(tmp_path):
    (tmp_path / "water.xyz").write_text("2\nEMT smoke test\nH 0 0 0\nH 0 0 1\n")
    (tmp_path / "input.json").write_text(
        json.dumps(
            {
                "input_structure_file": str(tmp_path / "water.xyz"),
                "output_results_file": str(tmp_path / "result.json"),
                "driver": "opt",
                "calculator": {"calculator_type": "emt"},
            }
        )
    )
    (tmp_path / "run_ase.py").write_bytes(
        (SKILLS / "chemgraph/scripts/run_ase.py").read_bytes()
    )
    environment = tmp_path / "environment script's.sh"
    environment.write_text(
        "printf ready > initialized\n"
        "export TMPDIR=/some/long/site/scratch/path\n"
    )
    python_wrapper = tmp_path / "python-wrapper.sh"
    python_wrapper.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$TMPDIR" "$http_proxy" "$https_proxy" > worker-environment.txt\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python_wrapper.chmod(0o755)
    script = (SKILLS / "pbs-hpc/assets/polaris-ase.pbs.template").read_text()
    for name, value in {
        "JOB_NAME": "cg-test",
        "PROJECT": "test",
        "FILESYSTEMS": "home:eagle",
        "ENVIRONMENT_FILE_SHELL": shlex.quote(str(environment)),
        "PYTHON_SHELL": shlex.quote(str(python_wrapper)),
    }.items():
        script = script.replace("{{" + name + "}}", value)
    batch = tmp_path / "job.pbs"
    batch.write_text(script)
    nodes = tmp_path / "nodes"
    nodes.write_text(socket.gethostname() + "\n")
    env = {
        **os.environ,
        "PBS_JOBID": "321.test",
        "PBS_NODEFILE": str(nodes),
        "PBS_O_WORKDIR": str(tmp_path),
    }
    bash = shutil.which("bash")
    subprocess.run([bash, "-n", str(batch)], check=True)
    result = subprocess.run([bash, str(batch)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "initialized").read_text() == "ready"
    assert (tmp_path / "worker-environment.txt").read_text().splitlines() == [
        "/tmp",
        "http://proxy.alcf.anl.gov:3128",
        "http://proxy.alcf.anl.gov:3128",
    ]
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["pbs_job_id"] == "321.test" and summary["converged"]
    assert (tmp_path / "final.xyz").exists()


@pytest.mark.skipif(os.name == "nt", reason="Polaris workers require a POSIX shell")
def test_parsl_template_initializes_compute_environment(tmp_path, monkeypatch):
    pytest.importorskip("parsl")
    import toml
    from chemgraph.hpc_configs.loader import load_parsl_config

    monkeypatch.delenv("PBS_JOBID", raising=False)
    environment = tmp_path / "environment script.sh"
    environment.write_text("export TMPDIR=/some/long/site/scratch/path\n")
    template = (SKILLS / "pbs-hpc/assets/polaris-parsl.toml.template").read_text()
    template = template.replace(
        "{{ENVIRONMENT_FILE_SHELL}}", shlex.quote(str(environment))
    )
    template = template.replace("{{PROJECT}}", "test").replace(
        "{{PARSL_RUN_DIR}}", str(tmp_path)
    )
    options = toml.loads(template)["execution"]["parsl"]
    config = load_parsl_config("polaris", address="127.0.0.1", **options)
    snippet = config.executors[0].provider.worker_init
    command = snippet + (
        '\nprintf "%s\\n" "$TMPDIR" "$http_proxy" "$https_proxy"\n'
    )
    result = subprocess.run(
        [shutil.which("bash"), "-c", command], capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == [
        "/tmp",
        "http://proxy.alcf.anl.gov:3128",
        "http://proxy.alcf.anl.gov:3128",
    ]


@pytest.mark.skipif(os.name == "nt", reason="PBS submission requires a POSIX shell")
@pytest.mark.parametrize("outcome", ["accepted", "uncertain", "empty"])
def test_direct_submission_retains_evidence_and_prevents_retry(tmp_path, outcome):
    (tmp_path / "input.json").write_text("{}")
    (tmp_path / "job.pbs").write_text("#!/bin/bash\ntrue\n")
    script = SKILLS / "pbs-hpc/scripts/submit_ase.sh"
    qsub = tmp_path / "qsub"
    qsub.write_text(
        "#!/bin/bash\necho called >> calls\n"
        + {
            "accepted": "echo 123.polaris.example\n",
            "uncertain": "echo connection-lost >&2\nexit 1\n",
            "empty": "exit 0\n",
        }[outcome]
    )
    qsub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
    }
    bash = shutil.which("bash")
    first = subprocess.run(
        [bash, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert first.returncode == (0 if outcome == "accepted" else 1)
    second = subprocess.run(
        [bash, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert second.returncode == 2
    assert (tmp_path / "calls").read_text().splitlines() == ["called"]
    assert (tmp_path / "submission.started").exists()


@pytest.mark.skipif(os.name == "nt", reason="PBS submission requires a POSIX shell")
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_agent_stages_batch_helper_and_preserves_submission_approval(
    tmp_path, decision
):
    from deepagents.backends import LocalShellBackend
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.types import Command
    from chemgraph.graphs.deep_agent import construct_deep_agent_graph
    from tests.test_deep_agent import _RecordingChatModel

    (tmp_path / "input.json").write_text("{}")
    (tmp_path / "job.pbs").write_text("#!/bin/bash\ntrue\n")
    qsub = tmp_path / "qsub"
    qsub.write_text("#!/bin/bash\necho called >> calls\necho 123.polaris.example\n")
    qsub.chmod(0o755)
    helper = (SKILLS / "pbs-hpc/scripts/submit_ase.sh").read_text()

    def call(name, args):
        return AIMessage(
            content="", tool_calls=[{"name": name, "args": args, "id": name}]
        )

    model = _RecordingChatModel(
        responses=[
            call(
                "read_file",
                {"file_path": "/chemgraph-skills/pbs-hpc/references/polaris-ase.md"},
            ),
            call(
                "read_file",
                {"file_path": "/chemgraph-skills/pbs-hpc/scripts/submit_ase.sh"},
            ),
            call(
                "write_file",
                {"file_path": "/workspace/submit_ase.sh", "content": helper},
            ),
            call(
                "execute",
                {"command": f"cd {shlex.quote(str(tmp_path))} && bash submit_ase.sh"},
            ),
            AIMessage(content="Submission reviewed."),
        ]
    )
    graph = construct_deep_agent_graph(
        model,
        discover_skills=False,
        backend=LocalShellBackend(
            root_dir=tmp_path,
            virtual_mode=True,
            inherit_env=False,
            env={"PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "")},
        ),
    )
    config = {"configurable": {"thread_id": "pbs-approval"}}
    state = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Prepare and submit the direct PBS example.")
            ]
        },
        config,
    )
    assert state["__interrupt__"] and not (tmp_path / "submit_ase.sh").exists()
    state = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert (tmp_path / "submit_ase.sh").read_text() == helper
    assert state["__interrupt__"] and not (tmp_path / "submission.started").exists()
    state = graph.invoke(Command(resume={"decisions": [{"type": decision}]}), config)
    assert "__interrupt__" not in state
    assert (tmp_path / "job.id").exists() == (decision == "approve")
    if decision == "approve":
        assert (tmp_path / "calls").read_text().splitlines() == ["called"]
