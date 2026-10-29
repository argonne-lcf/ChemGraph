import os
import subprocess
import sys
from pathlib import Path


def test_core_and_peer_messaging_import_without_academy():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = '''
import sys
class NoAcademy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "academy" or fullname.startswith("academy."):
            raise ImportError("Academy unavailable")
sys.meta_path.insert(0, NoAcademy())
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.academy import PeerMessage, make_peer_message_tool
from chemgraph.cli.main import main
'''
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": source, "PYTHONNOUSERSITE": "1"})
