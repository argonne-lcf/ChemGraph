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


@pytest.mark.parametrize("shell", ["/bin/sh", "bash", "dash"])
def test_every_documented_block_parses(shell):
    if not shutil.which(shell):
        pytest.skip(f"{shell} is not installed")
    for path in DOCS:
        for block in blocks(path):
            assert block.startswith("exec bash /dev/fd/3 3<<'BASH' </dev/null\n")
            assert block.endswith("BASH\n")
            subprocess.run([shell, "-n"], input=block, text=True, check=True,
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


def execute(root, block, backend):
    if backend:
        result = create_host_shell_backend(root).execute(block, timeout=30)
        assert result.exit_code == 0, result.output
        output = result.output
    else:
        result = subprocess.run(["/bin/sh", "-c", block], cwd=root, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        output = result.stdout
    assert "recipe-finished" in output
    return output


@pytest.mark.parametrize("backend", [False, True], ids=["shell", "deepagent"])
def test_actual_transfer_record_then_submit(workspace, backend):
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
    execute(workspace, recipe(path, "Record and submit a transfer", **values), backend)
    saved = json.loads((workspace / ".globus/run-1-stage-in.json").read_text())
    assert saved == {"submission_id": "sid-1", "source": values["SOURCE"],
                     "destination": values["DESTINATION"], "label": values["LABEL"],
                     "options": ["--recursive"]}
    execute(workspace, recipe(path, "Record and submit a transfer", 1, **values), backend)
    calls = [json.loads(line) for line in (workspace / "calls.jsonl").read_text().splitlines()]
    assert calls == [["task", "generate-submission-id"],
                     ["transfer", values["SOURCE"], values["DESTINATION"], "--recursive",
                      "--label", values["LABEL"], "--submission-id", "sid-1",
                      "--jmespath", "task_id", "--format", "unix"]]
    assert (workspace / ".globus/run-1-stage-in.id").read_text() == "task-1\n"


@pytest.mark.parametrize("backend", [False, True], ids=["shell", "deepagent"])
def test_actual_monitor_and_empty_listing(workspace, backend):
    (workspace / "target.toml").write_text(
        'compute_resource_id = "polaris"\nusername = "me"\nremote_root = "/eagle/P/runs"\n')
    (workspace / "run-1").mkdir()
    (workspace / "run-1/job.id").write_text("123.polaris\n")
    fake_cli(workspace, "alcf-tokens", '''import sys
assert sys.stdin.read() == ""
print("fake-token")
''')
    fake_cli(workspace, "curl", '''import json, pathlib, sys
assert sys.stdin.read() == "Authorization: Bearer fake-token\\n"
url = sys.argv[-1]
with pathlib.Path("urls.jsonl").open("a") as f:
    f.write(json.dumps(url) + "\\n")
if "offset=" in url:
    print("[]")
else:
    print(json.dumps({"id": "123.polaris", "status": {"state": "completed", "exit_code": 0}}))
''')
    path = DOCS[1]
    before = (workspace / "run-1/job.id").read_bytes()
    output = execute(workspace, recipe(path, "Monitor", RUN="run-1"), backend)
    assert '"state": "completed"' in output
    execute(workspace, recipe(path, "Monitor", 1, RUN="run-1", JOB="123.polaris"), backend)
    output = execute(workspace, recipe(path, "Inspect uncertainty", RUN="run-1"), backend)
    assert "all pages read" in output
    assert (workspace / "run-1/job.id").read_bytes() == before
    urls = [json.loads(line) for line in (workspace / "urls.jsonl").read_text().splitlines()]
    assert len(urls) == 3 and all("/compute/status/" in url for url in urls)
    assert urls[-1].endswith("historical=true&limit=100&offset=0")
