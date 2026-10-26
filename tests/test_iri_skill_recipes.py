"""Execute the documented Bash recipes through the same shells agents use."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

from chemgraph.agent.deepagent_backend import create_host_shell_backend

SKILLS = Path(__file__).resolve().parents[1] / "src/chemgraph/skills"
DOCS = [SKILLS / "iri-hpc/SKILL.md", SKILLS / "iri-hpc/references/api-v2.md",
        SKILLS / "globus-transfer/references/cli.md"]
pytestmark = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"),
                                reason="recipes require Bash and /dev/fd")


def blocks(path):
    return re.findall(r"```bash\n(.*?)```", path.read_text(), re.S)


def test_every_documented_block_parses():
    for path in DOCS:
        for block in blocks(path):
            assert block.startswith("exec bash /dev/fd/3 3<<'BASH' </dev/null\n")
            assert block.endswith("BASH\n")
            subprocess.run(["/bin/sh", "-n"], input=block, text=True, check=True,
                           capture_output=True)
            subprocess.run(["bash", "-n"], input="\n".join(block.splitlines()[1:-1]),
                           text=True, check=True, capture_output=True)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    if not shutil.which("jq"):
        pytest.skip("recipes require jq")
    root = tmp_path / "workspace with spaces"
    root.mkdir()
    (root / "bin").mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join([str(root / "bin"), str(Path(sys.executable).parent),
                                               os.environ["PATH"]]))
    monkeypatch.setenv("PYTHONPATH", str(SKILLS.parents[1]))
    return root


def fake_cli(root, name, body):
    path = root / "bin" / name
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


def recipe(path, heading, index=0, **values):
    text = path.read_text().split("## " + heading + "\n")[1].split("\n## ")[0]
    block = re.findall(r"```bash\n(.*?)```", text, re.S)[index]
    for key, value in values.items():
        block = re.sub(rf"^{key}=\.\.\..*$", lambda m: f"{key}={shlex.quote(value)}",
                       block, flags=re.M)
    return block.replace("\nBASH\n", "\n# Don't let a child consume this script (or its sentinel).\nprintf 'recipe-finished\\n'\nBASH\n")


def execute(backend, block):
    result = backend.execute(block, timeout=30)
    assert result.exit_code == 0, result.output
    assert "recipe-finished" in result.output


def test_actual_transfer_record_then_submit(workspace):
    backend = create_host_shell_backend(workspace)
    fake_cli(workspace, "globus", '''import json, pathlib, sys
assert sys.stdin.read() == "", "child must not consume the Bash script"
args = sys.argv[1:]
with pathlib.Path("calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\\n")
if args == ["task", "generate-submission-id"]:
    print("sid-1")
elif args[0] == "transfer":
    print("task-1")
else:
    raise SystemExit("unexpected Globus call")
''')
    path = DOCS[2]
    values = {"STATE": ".globus", "LABEL": "run-1-stage-in", "SOURCE": "L:/~/run 1/",
              "DESTINATION": "E:/P/run 1/"}
    execute(backend, recipe(path, "Record and submit a transfer", **values))
    saved = json.loads((workspace / ".globus/run-1-stage-in.json").read_text())
    assert saved == {"submission_id": "sid-1", "source": values["SOURCE"],
                     "destination": values["DESTINATION"], "label": values["LABEL"],
                     "options": ["--recursive"]}
    execute(backend, recipe(path, "Record and submit a transfer", 1, **values))
    calls = [json.loads(line) for line in (workspace / "calls.jsonl").read_text().splitlines()]
    assert calls == [["task", "generate-submission-id"],
                     ["transfer", values["SOURCE"], values["DESTINATION"], "--recursive",
                      "--label", values["LABEL"], "--submission-id", "sid-1",
                      "--jmespath", "task_id", "--format", "unix"]]
    assert (workspace / ".globus/run-1-stage-in.id").read_text() == "task-1\n"
