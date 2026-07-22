"""Pure-Python PACMOF2 charge-assignment helpers (no LangChain / MCP decorators).

PACMOF2 predicts partial atomic charges for MOFs using machine learning
(scikit-learn models). It is a standalone CIF-in/CIF-out engine, not an
ASE calculator. This module wraps ``pacmof2.get_charges`` and reads the
resulting charged CIF back into a compact summary.

``pacmof2`` (and its hard-pinned ``scikit-learn==1.3.2``) is imported
lazily inside :func:`run_pacmof2_core` so ChemGraph imports cleanly in
environments where PACMOF2 is not installed -- e.g. when the actual charge
assignment runs on a remote globus_compute endpoint with its own env.

Used by the LangChain ``@tool`` wrapper in :mod:`pacmof2_tools` and the
MCP wrapper in :mod:`chemgraph.mcp.pacmof2_mcp_hpc`.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from ase.io import read as ase_read

from chemgraph.schemas.pacmof2_schema import pacmof2_input_schema


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_atom_site_charges(cif_path: str) -> list[float]:
    """Fallback parser for the ``_atom_site_charge`` column of a CIF.

    Used when neither ASE nor pymatgen surface per-atom charges. Locates
    the ``_atom_site_charge`` position within the ``loop_`` header and
    reads that column from each atom-site row.
    """
    charges: list[float] = []
    with open(cif_path, "r") as fh:
        lines = fh.readlines()

    # Find the loop_ block whose headers include _atom_site_charge.
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            headers: list[str] = []
            j = i + 1
            while j < n and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            if "_atom_site_charge" in headers:
                charge_idx = headers.index("_atom_site_charge")
                for row in lines[j:]:
                    row = row.strip()
                    if not row or row.startswith("_") or row == "loop_":
                        break
                    fields = row.split()
                    if len(fields) <= charge_idx:
                        break
                    try:
                        charges.append(float(fields[charge_idx]))
                    except ValueError:
                        break
                return charges
            i = j
        else:
            i += 1
    return charges


def _read_pacmof2_output(
    output_cif: str,
    input_cif: Optional[str] = None,
    net_charge: Union[int, float, dict] = 0,
) -> dict:
    """Read a PACMOF2 output CIF and return a compact charge summary.

    Full per-atom charges remain in the output CIF; this returns only an
    aggregate summary for the agent.

    Parameters
    ----------
    output_cif : str
        Path to the charged CIF written by PACMOF2.
    input_cif : str, optional
        Path to the original input CIF (recorded for reference).
    net_charge : int | float | dict
        Target net charge requested for the assignment.

    Returns
    -------
    dict
        Summary including status, output path, per-element mean charges,
        and the sum of charges.
    """
    result = {
        "status": "failure",
        "output_cif_path": None,
        "input_cif_path": input_cif,
        "n_atoms": 0,
        "net_charge_input": net_charge,
        "sum_of_charges": None,
        "per_element_mean_charge": {},
        "charge_range": None,
    }

    if not Path(output_cif).exists():
        return result

    try:
        atoms = ase_read(output_cif)
        symbols = list(atoms.get_chemical_symbols())
        charges = list(atoms.get_initial_charges())

        # ASE may return an all-zero array when it doesn't parse the
        # _atom_site_charge column; fall back to a manual parse.
        if not any(charges):
            parsed = _parse_atom_site_charges(output_cif)
            if parsed:
                charges = parsed

        if not charges or len(charges) != len(symbols):
            raise ValueError(
                f"Could not read per-atom charges from {output_cif}"
            )

        per_element_sum: dict[str, float] = defaultdict(float)
        per_element_count: dict[str, int] = defaultdict(int)
        for sym, q in zip(symbols, charges):
            per_element_sum[sym] += q
            per_element_count[sym] += 1

        per_element_mean = {
            sym: round(per_element_sum[sym] / per_element_count[sym], 4)
            for sym in per_element_sum
        }

        result["status"] = "success"
        result["output_cif_path"] = str(Path(output_cif).resolve())
        result["n_atoms"] = len(symbols)
        result["sum_of_charges"] = round(float(sum(charges)), 4)
        result["per_element_mean_charge"] = per_element_mean
        result["charge_range"] = [
            round(float(min(charges)), 4),
            round(float(max(charges)), 4),
        ]
    except Exception as e:
        print(f"Error parsing PACMOF2 output in {output_cif}: {e}")
        result["status"] = "failure"

    return result


# ---------------------------------------------------------------------------
# Mock (for testing without PACMOF2 installed)
# ---------------------------------------------------------------------------


def mock_pacmof2(params: pacmof2_input_schema) -> dict:
    """Return mock PACMOF2 results for testing without the pacmof2 package.

    Parameters
    ----------
    params : pacmof2_input_schema
        Input parameters (used only for the output path and net charge).

    Returns
    -------
    dict
        A plausible charge summary matching the shape of the real result.
    """
    time.sleep(random.uniform(1, 3))

    cif_path = Path(params.input_structure_file)
    output_cif = cif_path.with_name(f"{cif_path.stem}{params.identifier}.cif")

    target = params.net_charge if isinstance(params.net_charge, (int, float)) else 0
    return {
        "status": "success",
        "output_cif_path": str(output_cif.resolve()),
        "input_cif_path": str(cif_path),
        "n_atoms": 4,
        "net_charge_input": params.net_charge,
        "sum_of_charges": round(float(target), 4),
        "per_element_mean_charge": {
            "O": round(random.uniform(-0.8, -0.4), 4),
            "Zn": round(random.uniform(0.6, 1.2), 4),
            "C": round(random.uniform(-0.2, 0.4), 4),
        },
        "charge_range": [-0.8, 1.2],
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_pacmof2_core(params: pacmof2_input_schema) -> dict:
    """Assign PACMOF2 partial atomic charges to a single MOF CIF.

    Parameters
    ----------
    params : pacmof2_input_schema
        Input parameters for the charge assignment.

    Returns
    -------
    dict
        Parsed charge summary from :func:`_read_pacmof2_output`.

    Raises
    ------
    FileNotFoundError
        If the input CIF does not exist.
    RuntimeError
        If the ``pacmof2`` package is not installed.
    """
    cif_path = Path(params.input_structure_file).resolve()
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file does not exist: {cif_path}")

    output_cif = cif_path.with_name(f"{cif_path.stem}{params.identifier}.cif")

    try:
        from pacmof2 import get_charges
    except ImportError as e:
        raise RuntimeError(
            "PACMOF2 is not installed. Install it from source "
            "(`pip install -e .` from github.com/snurr-group/pacmof2), or run "
            "this tool on a globus_compute endpoint whose environment has "
            "pacmof2 and scikit-learn==1.3.2."
        ) from e

    get_charges(
        path_to_cif=str(cif_path),
        output_path=str(output_cif.parent),
        identifier=params.identifier,
        multiple_cifs=False,
        adjust_charge_method=params.adjust_charge_method,
        net_charge=params.net_charge,
    )

    return _read_pacmof2_output(
        output_cif=str(output_cif),
        input_cif=str(cif_path),
        net_charge=params.net_charge,
    )
