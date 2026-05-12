"""Core PySCF helpers for ChemGraph MCP tools.

This module intentionally contains plain Python functions.  MCP wrappers live in
``chemgraph.mcp.mcp_tools`` and should delegate here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, get_args

import numpy as np

from chemgraph.schemas.pyscf_schema import (
    PySCFMolecularInput,
    PySCFPostHF,
    PySCFPeriodicInput,
    PySCFProperty,
    PySCFPropertyInput,
    PySCFReference,
    PySCFRecipeInput,
    StructureInput,
)

HARTREE_TO_EV = 27.211386245988


def _resolve_path(path: str) -> str:
    """Resolve paths under ``CHEMGRAPH_LOG_DIR`` when configured."""
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if log_dir and not os.path.isabs(path):
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, path)
    return path


def _pyscf_available() -> bool:
    return importlib.util.find_spec("pyscf") is not None


def _require_pyscf():
    if not _pyscf_available():
        raise ImportError(
            "PySCF is not installed. Install the optional PySCF dependencies, "
            "for example `pip install chemgraphagent[pyscf]` or `pip install pyscf`."
        )
    import pyscf

    return pyscf


def _to_builtin(value: Any) -> Any:
    """Convert numpy/PySCF-ish values to JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _json_dump(data: dict, output_json: str) -> str:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(data), f, indent=2, default=str)
    return str(path.resolve())


def _write_result(data: dict, output_json: str) -> str:
    """Write a result dict and include the artifact path in the file itself."""
    output_abs = str(Path(output_json).resolve())
    data.setdefault("artifacts", {})["output_json"] = output_abs
    return _json_dump(data, output_abs)


def _output_path(output_dir: str, output_json: str) -> str:
    resolved_dir = Path(_resolve_path(output_dir)).resolve()
    output_path = Path(output_json)
    if output_path.is_absolute():
        return str(output_path)
    return str(resolved_dir / output_path)


def _structure_to_atom_input(structure: StructureInput, unit: str):
    if structure.atom:
        return structure.atom

    if not structure.input_structure_file:
        raise ValueError("No structure source provided.")

    from ase.io import read as ase_read

    path = Path(structure.input_structure_file)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    atoms = ase_read(str(path), format=structure.fmt)
    atom_spec = [
        (symbol, tuple(float(x) for x in pos))
        for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.positions)
    ]
    return atom_spec


def _structure_cell_vectors(structure: StructureInput) -> Optional[list[list[float]]]:
    if structure.lattice_vectors is not None:
        return structure.lattice_vectors
    if not structure.input_structure_file:
        return None

    from ase.io import read as ase_read

    atoms = ase_read(structure.input_structure_file, format=structure.fmt)
    cell = atoms.cell.tolist()
    if not cell or not any(any(abs(x) > 1e-12 for x in row) for row in cell):
        return None
    return cell


def _energy_payload(energy_hartree: Optional[float]) -> dict:
    if energy_hartree is None:
        return {"hartree": None, "eV": None}
    return {
        "hartree": float(energy_hartree),
        "eV": float(energy_hartree) * HARTREE_TO_EV,
    }


def _build_molecule(params: PySCFMolecularInput | PySCFRecipeInput):
    from pyscf import gto

    atom = _structure_to_atom_input(params.structure, params.unit)
    return gto.M(
        atom=atom,
        basis=params.basis,
        unit=params.unit,
        charge=params.charge,
        spin=params.spin,
        verbose=params.verbose,
        max_memory=params.max_memory,
    )


def _build_scf_method(mol, params: PySCFMolecularInput | PySCFRecipeInput):
    from pyscf import dft, scf

    reference = params.reference.upper()
    if reference == "RHF":
        mf = scf.RHF(mol)
    elif reference == "UHF":
        mf = scf.UHF(mol)
    elif reference == "ROHF":
        mf = scf.ROHF(mol)
    elif reference == "RKS":
        mf = dft.RKS(mol)
        mf.xc = getattr(params, "xc", None) or "b3lyp"
    elif reference == "UKS":
        mf = dft.UKS(mol)
        mf.xc = getattr(params, "xc", None) or "b3lyp"
    else:
        raise ValueError(f"Unsupported molecular PySCF reference: {reference}")

    mf.max_cycle = params.max_cycle
    mf.conv_tol = params.conv_tol

    chkfile = getattr(params, "chkfile", None)
    if not chkfile:
        chkfile = str(Path(_resolve_path(params.output_dir)).resolve() / "pyscf.chk")
    chkfile_path = Path(chkfile)
    chkfile_path.parent.mkdir(parents=True, exist_ok=True)
    mf.chkfile = str(chkfile_path)
    return mf


def _run_post_hf(mf, methods: Iterable[str]) -> dict:
    results: Dict[str, dict] = {}
    ccsd_obj = None

    for method in methods:
        method_key = method.upper()
        if method_key == "MP2":
            mymp = mf.MP2()
            e_corr, _ = mymp.kernel()
            e_tot = getattr(mymp, "e_tot", None)
            if e_tot is None:
                e_tot = mf.e_tot + e_corr
            results["MP2"] = {
                "correlation_energy_hartree": float(e_corr),
                "total_energy": _energy_payload(float(e_tot)),
            }
        elif method_key == "CCSD":
            ccsd_obj = mf.CCSD()
            e_corr, _, _ = ccsd_obj.kernel()
            results["CCSD"] = {
                "correlation_energy_hartree": float(e_corr),
                "total_energy": _energy_payload(float(ccsd_obj.e_tot)),
                "converged": bool(getattr(ccsd_obj, "converged", False)),
            }
        elif method_key == "CCSD(T)":
            if ccsd_obj is None:
                ccsd_obj = mf.CCSD()
                ccsd_obj.kernel()
            triples_corr = ccsd_obj.ccsd_t()
            e_tot = ccsd_obj.e_tot + triples_corr
            results["CCSD(T)"] = {
                "triples_correction_hartree": float(triples_corr),
                "total_energy": _energy_payload(float(e_tot)),
            }
        else:
            raise ValueError(f"Unsupported post-HF method: {method}")

    return results


def _run_properties(mol, mf, properties: Iterable[str]) -> dict:
    results: Dict[str, Any] = {}

    for prop in properties:
        prop_key = prop.lower()
        if prop_key == "dipole":
            results["dipole"] = {
                "value": _to_builtin(mf.dip_moment(unit="Debye", verbose=0)),
                "unit": "Debye",
            }
        elif prop_key in {"population", "mulliken_population"}:
            pop, charges = mf.mulliken_pop(verbose=0)
            results["mulliken_population"] = {
                "population": _to_builtin(pop),
                "charges": _to_builtin(charges),
            }
        elif prop_key == "mo_energy":
            results["mo_energy"] = {
                "value": _to_builtin(getattr(mf, "mo_energy", None)),
                "unit": "Hartree",
            }
        elif prop_key == "gradient":
            grad = mf.nuc_grad_method().kernel()
            results["gradient"] = {
                "value": _to_builtin(grad),
                "unit": "Hartree/Bohr",
            }
        else:
            raise ValueError(f"Unsupported PySCF property: {prop}")

    return results


def _schema_literal_values(model_cls: type, field_name: str) -> list[str]:
    """Return string values from a Pydantic field annotated as a Literal."""
    return [
        str(value) for value in get_args(model_cls.model_fields[field_name].annotation)
    ]


def get_pyscf_capability_manifest_core() -> dict:
    """Return the supported PySCF MCP surface and current v0 limitations."""
    return {
        "status": "success",
        "manifest": "pyscf_capability_manifest",
        "pyscf_available": _pyscf_available(),
        "tools": {
            "run_pyscf_molecular": {
                "status": "implemented",
                "references": list(get_args(PySCFReference)),
                "post_hf": list(get_args(PySCFPostHF)),
                "properties": list(get_args(PySCFProperty)),
            },
            "run_pyscf_periodic": {
                "status": "minimal",
                "references": _schema_literal_values(PySCFPeriodicInput, "reference"),
            },
            "run_pyscf_recipe": {
                "status": "minimal",
                "recipes": _schema_literal_values(PySCFRecipeInput, "recipe"),
            },
            "run_pyscf_property": {
                "status": "implemented",
                "scope": "extract properties already stored in a PySCF result JSON",
            },
            "extract_pyscf_output": {
                "status": "implemented",
                "scope": "load a PySCF result JSON",
            },
        },
        "limitations": [
            "No arbitrary run_pyscf_code tool is exposed.",
            (
                "Solvent, QMMM, TDDFT, scans, and advanced active-space "
                "workflows are recipe candidates, not generic free-form execution."
            ),
            (
                "Periodic support is intentionally minimal and should be "
                "expanded with reference tests before production use."
            ),
        ],
    }


def run_pyscf_molecular_core(params: PySCFMolecularInput) -> dict:
    """Run a molecular PySCF calculation and write a JSON artifact."""
    started_at = time.time()
    output_json = _output_path(params.output_dir, params.output_json)

    if params.solvent is not None:
        raise NotImplementedError("solvent is reserved for a future PySCF recipe.")

    pyscf = _require_pyscf()
    mol = _build_molecule(params)
    mf = _build_scf_method(mol, params)
    energy = mf.kernel()

    post_hf_results = _run_post_hf(mf, params.post_hf)
    property_results = _run_properties(mol, mf, params.properties)

    result = {
        "status": "success",
        "calculation": "pyscf_molecular",
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "input": params.model_dump(),
        "molecule": {
            "natoms": int(mol.natm),
            "nelectron": int(mol.nelectron),
            "charge": int(mol.charge),
            "spin": int(mol.spin),
            "basis": params.basis,
            "unit": params.unit,
        },
        "scf": {
            "reference": params.reference,
            "xc": params.xc if params.reference in {"RKS", "UKS"} else None,
            "converged": bool(mf.converged),
            "total_energy": _energy_payload(float(energy)),
            "energy_unit": "Hartree",
            "cycles": getattr(mf, "cycles", None),
        },
        "post_hf": post_hf_results,
        "properties": property_results,
        "artifacts": {
            "chkfile": str(Path(mf.chkfile).resolve()) if mf.chkfile else None,
        },
        "wall_time": time.time() - started_at,
    }
    _write_result(result, output_json)
    return result


def run_pyscf_periodic_core(params: PySCFPeriodicInput) -> dict:
    """Run a minimal periodic PySCF HF/DFT calculation."""
    started_at = time.time()
    output_json = _output_path(params.output_dir, params.output_json)

    pyscf = _require_pyscf()
    from pyscf.pbc import dft, gto, scf

    atom = _structure_to_atom_input(params.structure, params.unit)
    lattice_vectors = _structure_cell_vectors(params.structure)
    if lattice_vectors is None:
        raise ValueError(
            "Periodic calculations require lattice_vectors or a structure "
            "file with a nonzero cell."
        )

    cell = gto.Cell()
    cell.atom = atom
    cell.a = lattice_vectors
    cell.basis = params.basis
    cell.pseudo = params.pseudo
    cell.unit = params.unit
    cell.charge = params.charge
    cell.spin = params.spin
    cell.verbose = params.verbose
    cell.max_memory = params.max_memory
    cell.build()

    reference = params.reference.upper()
    if reference.startswith("K"):
        kpts = cell.make_kpts(params.kpts)
        if reference == "KRHF":
            mf = scf.KRHF(cell, kpts=kpts)
        elif reference == "KUHF":
            mf = scf.KUHF(cell, kpts=kpts)
        elif reference == "KRKS":
            mf = dft.KRKS(cell, kpts=kpts)
            mf.xc = params.xc or "pbe"
        elif reference == "KUKS":
            mf = dft.KUKS(cell, kpts=kpts)
            mf.xc = params.xc or "pbe"
        else:
            raise ValueError(f"Unsupported periodic PySCF reference: {reference}")
    else:
        if reference == "RHF":
            mf = scf.RHF(cell)
        elif reference == "UHF":
            mf = scf.UHF(cell)
        elif reference == "RKS":
            mf = dft.RKS(cell)
            mf.xc = params.xc or "pbe"
        elif reference == "UKS":
            mf = dft.UKS(cell)
            mf.xc = params.xc or "pbe"
        else:
            raise ValueError(f"Unsupported periodic PySCF reference: {reference}")

    mf.max_cycle = params.max_cycle
    mf.conv_tol = params.conv_tol
    energy = mf.kernel()

    result = {
        "status": "success",
        "calculation": "pyscf_periodic",
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "input": params.model_dump(),
        "cell": {
            "natoms": int(cell.natm),
            "nelectron": int(cell.nelectron),
            "basis": params.basis,
            "pseudo": params.pseudo,
            "lattice_vectors": _to_builtin(lattice_vectors),
        },
        "scf": {
            "reference": reference,
            "xc": params.xc if "KS" in reference else None,
            "converged": bool(mf.converged),
            "total_energy": _energy_payload(float(energy)),
            "energy_unit": "Hartree",
        },
        "artifacts": {},
        "wall_time": time.time() - started_at,
    }
    _write_result(result, output_json)
    return result


def run_pyscf_property_core(params: PySCFPropertyInput) -> dict:
    """Extract stored properties from a PySCF result JSON."""
    started_at = time.time()

    with open(params.result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    stored = data.get("properties", {})
    if params.properties:
        selected = {}
        missing = []
        for prop in params.properties:
            key = "mulliken_population" if prop == "population" else prop
            if key in stored:
                selected[key] = stored[key]
            else:
                missing.append(prop)
        if missing:
            raise KeyError(
                "Requested properties were not found in the PySCF result JSON: "
                f"{missing}"
            )
    else:
        selected = stored
        missing = []

    result = {
        "status": "success",
        "calculation": "pyscf_property",
        "source_json": str(Path(params.result_json).resolve()),
        "properties": selected,
        "missing_properties": missing,
        "wall_time": time.time() - started_at,
        "artifacts": {},
    }

    if params.output_dir:
        output_json = _output_path(params.output_dir, params.output_json)
        _write_result(result, output_json)
    return result


def run_pyscf_recipe_core(params: PySCFRecipeInput) -> dict:
    """Run a whitelisted PySCF recipe."""
    started_at = time.time()
    output_json = _output_path(params.output_dir, params.output_json)

    pyscf = _require_pyscf()
    from pyscf import mcscf

    if params.recipe != "casscf_single_point":
        raise ValueError(f"Unsupported PySCF recipe: {params.recipe}")

    mol = _build_molecule(params)
    mf = _build_scf_method(mol, params)
    scf_energy = mf.kernel()

    cas = mcscf.CASSCF(
        mf,
        params.active_space.ncas,
        params.active_space.nelecas,
    )
    cas_energy = cas.kernel()[0]

    result = {
        "status": "success",
        "calculation": "pyscf_recipe",
        "recipe": params.recipe,
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "input": params.model_dump(),
        "scf": {
            "reference": params.reference,
            "converged": bool(mf.converged),
            "total_energy": _energy_payload(float(scf_energy)),
        },
        "casscf": {
            "converged": bool(getattr(cas, "converged", False)),
            "total_energy": _energy_payload(float(cas_energy)),
            "ncas": params.active_space.ncas,
            "nelecas": params.active_space.nelecas,
        },
        "artifacts": {
            "chkfile": str(Path(mf.chkfile).resolve()) if mf.chkfile else None,
        },
        "wall_time": time.time() - started_at,
    }
    _write_result(result, output_json)
    return result


def extract_pyscf_output_core(json_file: str) -> dict:
    """Load a saved PySCF JSON result."""
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)
