"""Execute skill examples and scripted agent actions without a live scheduler/LLM."""

import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

import pytest
from deepagents.backends import LocalShellBackend
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from tests.test_deep_agent import _RecordingChatModel, _message_content_text

SKILLS = Path(__file__).resolve().parents[1] / "src/chemgraph/skills"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="PBS uses a POSIX shell")


def example(path, language):
    return (SKILLS / path).read_text().split(f"```{language}\n", 1)[1].split("```", 1)[0]


@pytest.fixture
def job(tmp_path):
    script = example("chemgraph/references/ase-batch.md", "python")
    (tmp_path / "calculate.py").write_text(script)
    (tmp_path / "hydrogen.xyz").write_text("2\nEMT test\nH 0 0 0\nH 0 0 1\n")
    payload = {
        "input_structure_file": str(tmp_path / "hydrogen.xyz"),
        "output_results_file": str(tmp_path / "result.json"),
        "driver": "opt", "calculator": {"calculator_type": "emt"},
    }
    environment = tmp_path / "compute environment's.sh"
    environment.write_text('export CHEMGRAPH_LOG_DIR="$PWD"\n')
    batch = (SKILLS / "pbs-hpc/assets/job.pbs.template").read_text()
    for key, value in {
        "JOB_NAME": "test", "PROJECT": "test", "QUEUE": "debug", "NODES": "1",
        "SYSTEM": "polaris", "WALLTIME": "00:30:00", "FILESYSTEMS": "home:eagle",
        "ENVIRONMENT_SETUP": f"source {shlex.quote(str(environment))}",
        "APPLICATION_LAUNCH_COMMAND": f"exec {shlex.quote(sys.executable)} calculate.py",
    }.items():
        batch = batch.replace("{{" + key + "}}", value)
    (tmp_path / "job.pbs").write_text(batch)
    (tmp_path / "nodes").write_text(socket.gethostname() + "\n")
    env = {**os.environ, "PBS_JOBID": "123.test", "PBS_NODEFILE": str(tmp_path / "nodes"),
           "PBS_O_WORKDIR": str(tmp_path)}
    return payload, env


@pytest.mark.parametrize("outcome", ["opt", "vib", "nonconverged", "missing", "login", "wrong_host"])
def test_documented_calculation_and_batch(job, tmp_path, outcome):
    payload, env = job
    if outcome in {"opt", "vib"}:
        payload["driver"] = outcome
    elif outcome == "nonconverged":
        payload["steps"] = 0
    elif outcome == "missing":
        payload["input_structure_file"] = str(tmp_path / "absent.xyz")
    elif outcome == "login":
        env.pop("PBS_JOBID")
    else:
        (tmp_path / "nodes").write_text("different-compute-host\n")
    (tmp_path / "input.json").write_text(json.dumps(payload))
    result = subprocess.run(
        ["bash", "job.pbs"], cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    expected = 0 if outcome in {"opt", "vib"} else 2 if outcome == "nonconverged" else 1
    assert result.returncode == expected, result.stdout + result.stderr
    output = tmp_path / "result.json"
    if outcome in {"missing", "login", "wrong_host"}:
        assert not output.exists()
    else:
        data = json.loads(output.read_text())
        assert data["success"] and data["converged"] == (outcome != "nonconverged")
        assert isinstance(data["potential_energy"], float)
        assert (tmp_path / "hydrogen_opt.traj").exists()
        if outcome == "vib":
            assert data["vibrational_frequencies"]["frequencies"]
            assert (tmp_path / "frequencies_hydrogen.csv").exists()


@pytest.fixture
def scheduler(tmp_path):
    qsub = tmp_path / "qsub"
    qsub.write_text("#!/bin/bash\necho called >> calls\necho 123.test\n")
    qsub.chmod(0o755)
    qstat = tmp_path / "qstat"
    qstat.write_text('#!/bin/bash\ntest "$2" = 123.test || exit 1\necho "job_state = Q"\n')
    qstat.chmod(0o755)
    return {"PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "")}


@pytest.mark.parametrize("outcome", ["accepted", "uncertain", "empty"])
def test_documented_submission_preserves_evidence(scheduler, tmp_path, outcome):
    (tmp_path / "job.pbs").write_text("#!/bin/bash\ntrue\n")
    if outcome != "accepted":
        (tmp_path / "qsub").write_text(
            "#!/bin/bash\necho called >> calls\n" +
            ("echo connection-lost >&2\nexit 1\n" if outcome == "uncertain" else "exit 0\n")
        )
    command = example("pbs-hpc/SKILL.md", "bash")
    first = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=scheduler, capture_output=True)
    assert (first.returncode == 0) == (outcome == "accepted")
    assert (tmp_path / "submission.started").exists()
    evidence = {p: (tmp_path / p).read_bytes() for p in ("job.id", "qsub.stderr")}
    again = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=scheduler, capture_output=True)
    assert again.returncode != 0
    assert (tmp_path / "calls").read_text().splitlines() == ["called"]
    assert evidence == {p: (tmp_path / p).read_bytes() for p in evidence}


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_agent_writes_submits_and_new_session_monitors(job, scheduler, tmp_path, decision):
    payload, _ = job

    def call(name, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])

    def shell(command):
        return f"cd {shlex.quote(str(tmp_path))} && bash -c {shlex.quote(command)}"

    def graph(responses):
        return construct_deep_agent_graph(
            _RecordingChatModel(responses=responses), tools=[], discover_skills=False,
            backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True,
                                      inherit_env=False, env=scheduler),
        )

    responses = [call("read_file", file_path=f"/chemgraph-skills/{p}") for p in (
        "chemgraph/SKILL.md", "pbs-hpc/SKILL.md", "chemgraph/references/ase-batch.md",
    )]
    # Scripted model writes the example files through the real file tools.
    files = {name: (tmp_path / name).read_text() for name in ("calculate.py", "job.pbs")}
    files["input.json"] = json.dumps(payload)
    for name, content in files.items():
        (tmp_path / name).unlink(missing_ok=True)
        responses.append(call("write_file", file_path=f"/workspace/{name}", content=content))
    responses += [call("execute", command=shell(example("pbs-hpc/SKILL.md", "bash"))),
                  AIMessage(content="Submission reviewed.")]
    agent = graph(responses)
    config = {"configurable": {"thread_id": "submit"}}
    state = agent.invoke({"messages": [HumanMessage(content="Write and submit my PBS calculation.")]}, config)
    for name in files:
        assert state["__interrupt__"] and not (tmp_path / name).exists()
        state = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
        assert (tmp_path / name).read_text() == files[name]
    assert state["__interrupt__"] and not (tmp_path / "submission.started").exists()
    state = agent.invoke(Command(resume={"decisions": [{"type": decision}]}), config)
    assert "__interrupt__" not in state
    assert (tmp_path / "job.id").exists() == (decision == "approve")
    if decision == "reject":
        assert not (tmp_path / "calls").exists()
        return
    monitor = graph([
        call("read_file", file_path="/workspace/job.id"),
        call("execute", command=shell('qstat -f "$(cat job.id)"')),
        AIMessage(content="Job remains queued."),
    ])
    config = {"configurable": {"thread_id": "fresh-session"}}
    state = monitor.invoke({"messages": [HumanMessage(content="Inspect my existing job; do not resubmit.")]}, config)
    assert state["__interrupt__"]
    state = monitor.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert "__interrupt__" not in state
    outputs = "\n".join(_message_content_text(m) for m in state["messages"] if m.type == "tool")
    assert "123.test" in outputs and "job_state = Q" in outputs
    assert (tmp_path / "calls").read_text().splitlines() == ["called"]
