"""Opt-in real namespaces and networking. All API traffic terminates at local fixtures."""

import json
import os
from pathlib import Path
import socket
import sys
import tempfile

import pytest

from chemgraph.eval.config import BenchmarkConfig
from chemgraph.eval.podman import PodmanRuntime

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("CHEMGRAPH_TEST_PODMAN") != "1",
    reason="Set CHEMGRAPH_TEST_PODMAN=1 with the three prebuilt test images",
)


@pytest.mark.asyncio
async def test_isolation_network_and_fresh_workers(tmp_path, monkeypatch):
    worker = os.environ["CHEMGRAPH_TEST_PODMAN_IMAGE"]
    proxy = os.environ["CHEMGRAPH_TEST_PODMAN_PROXY_IMAGE"]
    service = os.environ["CHEMGRAPH_TEST_PODMAN_SERVICE_IMAGE"]
    # Plant sentinels in the controller's dataset, previous output, and actual home.
    with tempfile.NamedTemporaryFile(prefix=".chemgraph-leakage-", dir=Path.home(), delete=False) as marker:
        home_marker = Path(marker.name)
    hidden = [tmp_path / "answers.json", tmp_path / "old-results/answer.txt", home_marker]
    for path in hidden:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("HOST_ONLY_SENTINEL")
    hidden[0].write_text(json.dumps([{"id": "3", "query": "Find water", "answer": "HOST_ONLY_SENTINEL"}]))
    config = BenchmarkConfig(
        models=["argo:gpt-5.6-sol"], workflow_types=["deep_agent"], dataset=str(hidden[0]),
        output_dir=str(tmp_path / "results"), judge_type="structured", deepagent_auto_approve=True,
        eval_runtime="podman", podman_image=worker, podman_proxy_image=proxy,
    )
    monkeypatch.setenv("ARGO_USER", "offline-fixture")
    runtime = PodmanRuntime(config)
    original = runtime.command
    fixture_name = f"cg-eval-{runtime.run_id}-services"

    async def command(*args, **kwargs):
        result = await original(*args, **kwargs)
        if args[:2] == ("network", "create") and args[-1] == runtime.outbound:
            runtime.resources.append(("container", fixture_name))
            # Only the proxy can reach this network. Both real names resolve to the fixture.
            await original("run", "--detach", "--pull=never", "--http-proxy=false", "--name", fixture_name,
                           "--label", runtime.label, "--network", runtime.outbound,
                           "--network-alias=apps.inside.anl.gov", "--network-alias=pubchem.ncbi.nlm.nih.gov", service)
        return result

    monkeypatch.setattr(runtime, "command", command)
    listener = socket.socket()
    listener.bind(("0.0.0.0", 0))
    listener.listen()
    try:
        await runtime.prepare()  # Tests cgroups, TLS, DNS, routes and proxy-denial rules against fixtures.
        first, first_volume = await runtime.new_worker()
        # A sibling exists on the same network with private data and a running process.
        sibling_script = "from pathlib import Path; import time; Path('/workspace/sibling-answer').write_text('SIBLING_SECRET'); time.sleep(120)"
        sibling_args = runtime.worker_args(first, first_volume)
        sibling_args.insert(1, "--detach")
        await runtime.command(*sibling_args, "-c", sibling_script)
        second, second_volume = await runtime.new_worker()
        script = '''
import os, pathlib, socket, subprocess
from ase.build import molecule
from ase.calculators.emt import EMT
from chemgraph.tools.ase_tools import file_to_atomsdata
from deepagents.backends import LocalShellBackend
import requests
hidden = HIDDEN
backend = LocalShellBackend(root_dir='/workspace', virtual_mode=False, inherit_env=False)
for target in hidden + ['/workspace/sibling-answer', '/proc/1/root' + hidden[0]]:
    assert not pathlib.Path(target).exists(), target
    assert backend.read(target).error, target
    assert subprocess.run(['test', '-r', target]).returncode != 0, target
    link = pathlib.Path('/workspace/escape')
    link.symlink_to(target)
    assert not link.exists()
    try:
        file_to_atomsdata.invoke({'fname': str(link)})
    except Exception:
        pass
    else:
        raise AssertionError('Chemistry reader escaped workspace')
    link.unlink()
for process in pathlib.Path('/proc').iterdir():
    if process.name.isdigit() and int(process.name) != os.getpid():
        try:
            assert b'SIBLING_SECRET' not in (process / 'cmdline').read_bytes()
        except (FileNotFoundError, PermissionError):
            pass
# Confirm proxy variables cannot create a direct route, including to host services.
for host, port in [('1.1.1.1', 443), ('10.0.2.2', PORT), ('127.0.0.1', PORT)]:
    try:
        connection = socket.create_connection((host, port), timeout=1)
    except OSError:
        pass
    else:
        connection.close()
        raise AssertionError('Direct connection succeeded')
assert requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/water/property/CanonicalSMILES/JSON', timeout=10).json()['PropertyTable']
try:
    session = requests.Session()
    session.trust_env = False
    session.get('https://pubchem.ncbi.nlm.nih.gov/', timeout=2)
except requests.RequestException:
    pass
else:
    raise AssertionError('Proxy bypass succeeded')
atoms = molecule('H2O')
atoms.calc = EMT()
assert isinstance(atoms.get_potential_energy(), float)
pathlib.Path('/workspace/own-artifact').write_text('own')
print('isolation-ok')
'''.replace("HIDDEN", repr([str(p) for p in hidden])).replace("PORT", str(listener.getsockname()[1]))
        output = await runtime.command(*runtime.worker_args(second, second_volume), "-c", script, timeout=60)
        assert b"isolation-ok" in output
        await runtime.remove("container", second)
        await runtime.remove("volume", second_volume)
        await runtime.remove("container", first)
        await runtime.remove("volume", first_volume)
        # Interrupt a real worker, then verify a fresh worker can complete normally.
        config.query_timeout_seconds = 0.01
        interrupted = await runtime.run_query(config.models[0], "Return water's SMILES.", "interrupted")
        assert interrupted["status"] == "timeout"
        config.query_timeout_seconds = 1800
        # Exercise the real single-query worker and formatter with a fake Argo service.
        raw = await runtime.run_query(config.models[0], "Return water's SMILES.", "3")
        assert raw["status"] == "completed", raw
        assert raw["structured_output"]["smiles"] == ["O"]
        assert raw["workspace"] == "/workspace"
        assert Path(raw["artifact_location"]).is_dir()
    finally:
        listener.close()
        home_marker.unlink(missing_ok=True)
        await runtime.close()
    assert runtime.resources == []
