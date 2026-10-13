import os
import subprocess
import sys
from pathlib import Path


def test_dashboard_and_cli_trace_without_academy():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = '''
import sys
class NoAcademy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "academy" or fullname.startswith("academy.") or fullname.startswith("chemgraph.academy"):
            raise ImportError("Academy unavailable")
sys.meta_path.insert(0, NoAcademy())
from chemgraph.cli.trace import CLIRunTrace
from chemgraph.distributed import PeerRuntime
from chemgraph.distributed.cli import main
from chemgraph.distributed.config import DistributedConfig
from chemgraph.dashboard.server import _static_file
assert b"<html" in _static_file("index.html", "text/html")[0].lower()
assert _static_file("app.js", "text/javascript")[0]
'''
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": source, "PYTHONNOUSERSITE": "1"})
