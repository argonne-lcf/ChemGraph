"""Prepare weights and record the exact installed image contents at build time."""

import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform

from ase.build import molecule
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from tblite.ase import TBLite
import torch

assert torch.version.cuda is None, "The evaluation image must use CPU PyTorch"
atoms = molecule("H2O")
atoms.calc = MaceCalc(calculator_type="mace_mp", model="medium-mpa-0", device="cpu").get_calculator()
mace_energy = float(atoms.get_potential_energy())
atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
tblite_energy = float(atoms.get_potential_energy())
cache = Path("/opt/model-cache")
weights = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in cache.rglob("*") if path.is_file()}
assert weights, "MACE weights were not cached in the image"
Path("/opt/chemgraph-evaluation-manifest.json").write_text(json.dumps({
    "version": 1, "python": platform.python_version(), "model": "medium-mpa-0", "weights": weights,
    "dependencies": {dist.metadata["Name"]: dist.version for dist in distributions()},
    "smoke_energies_eV": {"mace": mace_energy, "tblite": tblite_energy},
}, indent=2, sort_keys=True))
