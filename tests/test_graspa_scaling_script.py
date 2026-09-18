"""Hermetic native scaling client and Aurora service lifecycle coverage."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import asynccontextmanager

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/graspa_scaling/run.sh"
spec = importlib.util.spec_from_file_location("graspa_scaling", ROOT / "scripts/graspa_scaling/run_graspa.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def batch(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Aurora launcher requires Linux, Bash 4.3+, and GNU coreutils")
    stub = tmp_path / "python"
    stub.write_text(f"#!{sys.executable}\n" + r"""
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(os.environ["CG_TEST_DIR"])
args = sys.argv[1:]
if args[0] == "-m":
    (root / "server.pid").write_text(str(os.getpid()))
    def stop(*_):
        (root / "server.stopped").touch()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    while True:
        time.sleep(0.05)
elif args[0] == "-" and args[1].startswith("http"):
    sys.stdin.read()
    deadline = time.monotonic() + 5
    while not (root / "server.pid").exists():
        if time.monotonic() > deadline:
            raise SystemExit("Server was not started before readiness")
        time.sleep(0.01)
    raise SystemExit(int(os.environ.get("CG_TEST_READY_EXIT", "0")))
elif args[0] in {"-", "-c"}:
    if args[0] == "-":
        sys.stdin.read()
else:
    (root / "client.json").write_text(json.dumps({
        "args": args,
        "backend": os.environ["CHEMGRAPH_EXECUTION_BACKEND"],
        "system": os.environ["COMPUTE_SYSTEM"],
        "worker_init": os.environ["CHEMGRAPH_WORKER_INIT"],
        "pythonpath": os.environ["PYTHONPATH"],
        "no_proxy": os.environ["NO_PROXY"],
    }))
    raise SystemExit(int(os.environ.get("CG_TEST_CLIENT_EXIT", "0")))
""")
    stub.chmod(0o755)
    venv = tmp_path / "env with spaces"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("# Prepared environment stub\n")
    nodefile = tmp_path / "nodes"
    nodefile.write_text("allocated-node\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CG_", "CHEMGRAPH_", "PBS_", "ADS_", "DES_"))
           and key not in {"ALCF_ACCESS_TOKEN", "N_CYCLES"}}
    env.update(PBS_NODEFILE=str(nodefile), PBS_O_WORKDIR=str(ROOT), PBS_JOBID="test-job",
               CG_ENV=str(venv), CG_MODULES="", CG_PYTHON=str(stub), CG_MODEL="fake",
               CHEMGRAPH_GRASPA_EXECUTABLE=str(stub), CG_RUN_DIR=str(tmp_path / "output"),
               CG_TEST_DIR=str(tmp_path), CG_STARTUP_TIMEOUT="10", CG_AGENT_TIMEOUT="10")

    def run(**overrides):
        return subprocess.run(["bash", str(LAUNCHER)], env={**env, **overrides},
                              text=True, capture_output=True, timeout=20)

    return run, tmp_path, env


def assert_server_stopped(root):
    assert (root / "server.stopped").exists()
    pid = int((root / "server.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def inputs(tmp_path):
    directory = tmp_path / "cifs"
    directory.mkdir()
    for name in ("b.CIF", "a.cif", "ignored.txt"):
        (directory / name).touch()
    return directory


def test_reference_conditions_subset_and_original_identity(tmp_path):
    directory = inputs(tmp_path)
    args = runner.parse_args(["--input-dir", str(directory), "--output-dir", str(tmp_path / "out"), "--limit", "1"])
    root, sources, manifest = runner.prepare(args)
    assert manifest["structures"] == 1
    assert manifest["simulations"] == 2
    assert sources == [str(directory / "a.cif")]
    assert manifest["conditions"] == [{"temperature": 298, "pressure": 960}, {"temperature": 298, "pressure": 320}]
    assert manifest["n_cycles_per_phase"] == 2_000_000
    assert not root.exists()
    query = runner.prepare_query(root, sources, manifest, False)
    assert str(root / "inputs") in query
    assert str(directory / "a.cif") not in query
    from chemgraph.mcp.graspa_mcp_hpc import _local_structure_files
    assert _local_structure_files(str(root / "inputs")) == sources
    args.resume = True
    (directory / "a.cif").unlink()
    assert runner.prepare(args) == (root, sources, manifest)
    assert runner.prepare_query(root, sources, manifest, True) == query


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "partial", "failed", "incomplete"])
async def test_runner_uses_native_graph_and_resumes(tmp_path, monkeypatch, status):
    from tests.test_graspa_graph import Model
    from tests.test_graspa_workflow import Backend
    from tests.test_graphs import _fake_prepared

    directory = inputs(tmp_path)
    args = runner.parse_args(["--input-dir", str(directory), "--output-dir", str(tmp_path / "out")])
    root, sources, manifest = runner.prepare(args)
    model, backend = Model(root / "inputs"), Backend(tmp_path)
    if status == "partial":
        backend.failed_sources.add(sources[0])
    elif status == "failed":
        backend.fail_all = True
    elif status == "incomplete":
        backend.mutate = lambda rows: rows[:-1]

    class Client:
        def __init__(self, _config):
            pass

        @asynccontextmanager
        async def session(self, _name):
            yield backend

    async def load(_session):
        return backend.tools()

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)
    monkeypatch.setattr("langchain_mcp_adapters.tools.load_mcp_tools", load)
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", lambda **kw: (model, _fake_prepared()[1]))
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(root))
    expected = 0 if status == "completed" else 1
    assert await runner.run(args, root, sources, manifest) == expected
    assert backend.calls["run_graspa_ensemble"] == 1
    assert json.loads((root / "analysis.json").read_text())["status"] == status
    journal = json.loads((root / "workflow.json").read_text())
    assert len(journal["requests"]) == 1
    assert journal["requests"]["task_1"]["input_structures"] == sources
    assert journal["requests"]["task_1"]["conditions"] == manifest["conditions"]
    args.resume = True
    assert await runner.run(args, *runner.prepare(args)) == expected
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.parametrize("client_exit", [0, 7])
def test_scaling_shell_worker_environment_and_cleanup(batch, client_exit):
    run, root, _env = batch
    directory = inputs(root)
    result = run(CG_CIF_DIR=str(directory), OMP_NUM_THREADS="208", CG_TEST_CLIENT_EXIT=str(client_exit))
    assert result.returncode == client_exit, result.stdout + result.stderr
    data = json.loads((root / "client.json").read_text())
    assert data["args"][0] == str(ROOT / "scripts/graspa_scaling/run_graspa.py")
    for flag, value in (("--ads-temp", "298"), ("--ads-pressure", "960"),
                        ("--des-pressure", "320"), ("--n-cycles", "2000000")):
        assert data["args"][data["args"].index(flag) + 1] == value
    assert "OMP_NUM_THREADS=1 " in data["worker_init"]
    initialized = subprocess.run(["bash", "-c", data["worker_init"]], text=True, capture_output=True)
    assert initialized.returncode == 0, initialized.stderr
    assert "127.0.0.1" in data["no_proxy"]
    assert_server_stopped(root)


def test_readiness_failure_stops_server(batch):
    run, root, _env = batch
    result = run(CG_CIF_DIR=str(inputs(root)), CG_TEST_READY_EXIT="3")
    assert result.returncode == 3, result.stdout + result.stderr
    assert not (root / "client.json").exists()
    assert_server_stopped(root)


def test_missing_allocation_starts_nothing(batch):
    run, root, _env = batch
    result = run(PBS_NODEFILE="")
    assert result.returncode == 2
    assert not (root / "server.pid").exists()
