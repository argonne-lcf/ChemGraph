"""Lazy agent wrappers for local molecular property workflows."""

from langchain_core.tools import tool
from chemgraph.tools import molecular_core


@tool
def prepare_conformers(
    smiles: str,
    output_dir: str = "molecular_properties",
    charge: int | None = None,
    candidates: int = 20,
    retained: int = 5,
    seed: int = 2025,
    rmsd_threshold: float = 0.5,
    mmff_max_iters: int = 2000,
) -> dict:
    """Prepare specified-stereo closed-shell molecular conformers with ETKDGv3/MMFF94s.

    Writes SDF/XYZ and a manifest in a fresh directory. Local unscoped only.
    Set retained <= candidates. All nonconverged candidates are excluded.
    """
    return molecular_core.prepare_conformers(
        smiles,
        output_dir,
        charge,
        candidates,
        retained,
        seed,
        rmsd_threshold,
        mmff_max_iters,
    )


@tool
def calculate_reactivity_descriptors(result_file: str) -> dict:
    """Read saved electronic results and estimate global reactivity from orbitals.

    Uses eta=I-A; these are orbital estimates, not experimental IP/EA or
    atom-specific reactivity. No installed engine is required. Local unscoped only.
    """
    return molecular_core.calculate_reactivity_descriptors(result_file)
