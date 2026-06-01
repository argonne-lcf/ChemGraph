"""Core PySCF helpers for ChemGraph MCP tools.

This module intentionally contains plain Python functions. MCP wrappers live in
``chemgraph.mcp.mcp_tools`` and should delegate here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from chemgraph.schemas.pyscf_schema import (
    PySCFCrystalReference,
    PySCFCrystalSpec,
    PySCFDevice,
    PySCFDriver,
    PySCFMoleculeReference,
    PySCFMoleculeSpec,
    PySCFUnit,
)

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

_MOLECULE_DRIVER_ALIASES = {
    "energy": "energy",
    "single_point": "energy",
    "sp": "energy",
    "optimization": "optimization",
    "opt": "optimization",
    "vibration": "vibration",
    "vib": "vibration",
    "thermochemistry": "thermochemistry",
    "thermo": "thermochemistry",
}


def _resolve_path(path: str) -> str:
    """Resolve relative paths under ``CHEMGRAPH_LOG_DIR`` when configured."""
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


def _apply_pyscf_device(obj: Any, device: str, label: str):
    """Apply the requested PySCF execution device to a PySCF object."""
    if device == "cpu":
        return obj

    if device == "gpu":
        if importlib.util.find_spec("gpu4pyscf") is None:
            raise ImportError(
                "PySCF GPU execution requires gpu4pyscf, but gpu4pyscf is "
                "not installed. Install a gpu4pyscf package compatible with "
                "your CUDA stack, then retry with device='gpu'."
            )

        to_gpu = getattr(obj, "to_gpu", None)
        if not callable(to_gpu):
            raise NotImplementedError(
                f"PySCF GPU execution is not available for {label}; the "
                "object does not expose a callable .to_gpu() method."
            )
        return to_gpu()

    raise ValueError(f"Unsupported PySCF device: {device}")


def _to_builtin(value: Any) -> Any:
    """Convert numpy/PySCF-ish values to JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if type(value).__module__.startswith("cupy") and hasattr(value, "get"):
        return _to_builtin(value.get())
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _json_dump(data: dict, output_json: str) -> str:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(data), f, indent=2, default=str)
    return str(path.resolve())


def _write_result(data: dict, output_json: Optional[str]) -> Optional[str]:
    if not output_json:
        return None
    output_abs = str(Path(_resolve_path(output_json)).resolve())
    data.setdefault("artifacts", {})["output_json"] = output_abs
    return _json_dump(data, output_abs)


def _read_json_artifact(json_file: str) -> dict:
    path = Path(_resolve_path(json_file)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PySCF JSON artifact not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"PySCF JSON artifact must contain a JSON object: {path}")
    return data


def _read_ase_structure(structure_file: str, fmt: Optional[str] = None):
    from ase.io import read as ase_read

    path = Path(_resolve_path(structure_file)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")
    return ase_read(str(path), format=fmt), str(path.resolve())


def _ase_atoms_payload(atoms) -> dict:
    return {
        "symbols": list(atoms.get_chemical_symbols()),
        "positions": _to_builtin(np.asarray(atoms.get_positions(), dtype=float)),
        "cell": _to_builtin(np.asarray(atoms.cell.array, dtype=float)),
        "pbc": [bool(x) for x in atoms.pbc],
        "formula": atoms.get_chemical_formula(),
    }


def _energy_payload(energy_hartree: Optional[float]) -> dict:
    if energy_hartree is None:
        return {"hartree": None, "eV": None}
    return {
        "hartree": float(energy_hartree),
        "eV": float(energy_hartree) * HARTREE_TO_EV,
    }


def _energy_payload_from_ev(energy_ev: Optional[float]) -> dict:
    if energy_ev is None:
        return {"hartree": None, "eV": None}
    return {
        "hartree": float(energy_ev) / HARTREE_TO_EV,
        "eV": float(energy_ev),
    }


def _normalise_driver(driver: str) -> str:
    normalised = _MOLECULE_DRIVER_ALIASES.get(driver.lower())
    if normalised is None:
        allowed = ", ".join(sorted(_MOLECULE_DRIVER_ALIASES))
        raise ValueError(
            f"Unsupported PySCF driver '{driver}'. Allowed values: {allowed}"
        )
    return normalised


def _coerce_molecule_spec(pyscf_molecule: Mapping[str, Any] | PySCFMoleculeSpec):
    if isinstance(pyscf_molecule, PySCFMoleculeSpec):
        return pyscf_molecule
    data = dict(pyscf_molecule)
    if "pyscf_molecule" in data:
        data = data["pyscf_molecule"]
    return PySCFMoleculeSpec.model_validate(data)


def _coerce_crystal_spec(pyscf_crystal: Mapping[str, Any] | PySCFCrystalSpec):
    if isinstance(pyscf_crystal, PySCFCrystalSpec):
        return pyscf_crystal
    data = dict(pyscf_crystal)
    if "pyscf_crystal" in data:
        data = data["pyscf_crystal"]
    return PySCFCrystalSpec.model_validate(data)


def _resolve_molecule_spec(
    pyscf_molecule: Mapping[str, Any] | PySCFMoleculeSpec | None,
    pyscf_molecule_json: Optional[str],
) -> tuple[PySCFMoleculeSpec, Optional[str]]:
    if pyscf_molecule is None:
        if not pyscf_molecule_json:
            raise ValueError(
                "run_pyscf_molecule requires either pyscf_molecule or "
                "pyscf_molecule_json."
            )
        loaded = _read_json_artifact(pyscf_molecule_json)
        return _coerce_molecule_spec(loaded), str(
            Path(_resolve_path(pyscf_molecule_json)).expanduser().resolve()
        )
    return _coerce_molecule_spec(pyscf_molecule), None


def _resolve_crystal_spec(
    pyscf_crystal: Mapping[str, Any] | PySCFCrystalSpec | None,
    pyscf_crystal_json: Optional[str],
) -> tuple[PySCFCrystalSpec, Optional[str]]:
    if pyscf_crystal is None:
        if not pyscf_crystal_json:
            raise ValueError(
                "run_pyscf_crystal requires either pyscf_crystal or "
                "pyscf_crystal_json."
            )
        loaded = _read_json_artifact(pyscf_crystal_json)
        return _coerce_crystal_spec(loaded), str(
            Path(_resolve_path(pyscf_crystal_json)).expanduser().resolve()
        )
    return _coerce_crystal_spec(pyscf_crystal), None


def _atom_tuples(symbols: list[str], positions: list[list[float]]):
    return [
        (symbol, tuple(float(x) for x in position))
        for symbol, position in zip(symbols, positions)
    ]


def _build_molecule(
    spec: PySCFMoleculeSpec,
    positions: Optional[list[list[float]]] = None,
):
    from pyscf import gto

    atom_positions = positions if positions is not None else spec.positions
    return gto.M(
        atom=_atom_tuples(spec.symbols, atom_positions),
        basis=spec.basis,
        unit=spec.unit,
        charge=spec.charge,
        spin=spec.spin,
        verbose=spec.verbose,
        max_memory=spec.max_memory,
    )


def _build_molecule_scf(
    spec: PySCFMoleculeSpec,
    mol,
    *,
    max_cycle: int,
    conv_tol: float,
    chkfile: Optional[str] = None,
):
    from pyscf import dft, scf

    reference = spec.reference.upper()
    if reference == "RHF":
        mf = scf.RHF(mol)
    elif reference == "UHF":
        mf = scf.UHF(mol)
    elif reference == "ROHF":
        mf = scf.ROHF(mol)
    elif reference == "RKS":
        mf = dft.RKS(mol)
        mf.xc = spec.xc or "b3lyp"
    elif reference == "UKS":
        mf = dft.UKS(mol)
        mf.xc = spec.xc or "b3lyp"
    else:
        raise ValueError(f"Unsupported molecular PySCF reference: {reference}")

    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol
    if chkfile:
        chkfile_path = Path(_resolve_path(chkfile)).resolve()
        chkfile_path.parent.mkdir(parents=True, exist_ok=True)
        mf.chkfile = str(chkfile_path)
    return mf


def _build_cell(
    spec: PySCFCrystalSpec,
    positions: Optional[list[list[float]]] = None,
):
    from pyscf.pbc import gto

    atom_positions = positions if positions is not None else spec.positions
    cell = gto.Cell()
    cell.atom = _atom_tuples(spec.symbols, atom_positions)
    cell.a = spec.lattice_vectors
    cell.basis = spec.basis
    cell.pseudo = spec.pseudo
    cell.unit = spec.unit
    cell.charge = spec.charge
    cell.spin = spec.spin
    cell.verbose = spec.verbose
    cell.max_memory = spec.max_memory
    cell.build()
    return cell


def _build_crystal_scf(
    spec: PySCFCrystalSpec,
    cell,
    *,
    max_cycle: int,
    conv_tol: float,
):
    from pyscf.pbc import dft, scf

    reference = spec.reference.upper()
    if reference.startswith("K"):
        kpts = cell.make_kpts(spec.kpts)
        if reference == "KRHF":
            mf = scf.KRHF(cell, kpts=kpts)
        elif reference == "KUHF":
            mf = scf.KUHF(cell, kpts=kpts)
        elif reference == "KRKS":
            mf = dft.KRKS(cell, kpts=kpts)
            mf.xc = spec.xc or "pbe"
        elif reference == "KUKS":
            mf = dft.KUKS(cell, kpts=kpts)
            mf.xc = spec.xc or "pbe"
        else:
            raise ValueError(f"Unsupported periodic PySCF reference: {reference}")
    else:
        if reference == "RHF":
            mf = scf.RHF(cell)
        elif reference == "UHF":
            mf = scf.UHF(cell)
        elif reference == "RKS":
            mf = dft.RKS(cell)
            mf.xc = spec.xc or "pbe"
        elif reference == "UKS":
            mf = dft.UKS(cell)
            mf.xc = spec.xc or "pbe"
        else:
            raise ValueError(f"Unsupported periodic PySCF reference: {reference}")

    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol
    return mf


def _atoms_from_molecule_spec(spec: PySCFMoleculeSpec):
    from ase import Atoms

    return Atoms(symbols=spec.symbols, positions=spec.positions)


def _atoms_from_crystal_spec(spec: PySCFCrystalSpec):
    from ase import Atoms

    return Atoms(
        symbols=spec.symbols,
        positions=spec.positions,
        cell=spec.lattice_vectors,
        pbc=spec.pbc,
    )


def _is_linear_atoms(atoms, tol: float = 1e-3) -> bool:
    if len(atoms) <= 2:
        return len(atoms) == 2
    coords = np.asarray(atoms.get_positions(), dtype=float)
    centered = coords - np.mean(coords, axis=0)
    _, singular_values, _ = np.linalg.svd(centered)
    if singular_values[0] == 0:
        return False
    return bool((singular_values[1] / singular_values[0]) < tol)


def _run_optimizer(atoms, optimizer: str, fmax: float, steps: int) -> bool:
    from ase.optimize import BFGS, FIRE, LBFGS, MDMin

    optimizers = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "fire": FIRE,
        "mdmin": MDMin,
    }
    optimizer_cls = optimizers.get(optimizer.lower())
    if optimizer_cls is None:
        raise ValueError(
            "Unsupported optimizer: "
            f"{optimizer}. Allowed values: {', '.join(sorted(optimizers))}"
        )
    if len(atoms) <= 1:
        return True
    dyn = optimizer_cls(atoms, logfile=None)
    return bool(dyn.run(fmax=fmax, steps=steps))


def _run_vibrations(atoms, displacement: float) -> dict:
    from ase import units
    from ase.vibrations import Vibrations

    with tempfile.TemporaryDirectory(prefix="chemgraph_pyscf_vib_") as tmpdir:
        vib = Vibrations(atoms, name=os.path.join(tmpdir, "vib"), delta=displacement)
        vib.clean()
        vib.run()
        energies = vib.get_energies()

    frequencies_cm = [energy / units.invcm for energy in energies]
    return {
        "n_modes": len(energies),
        "frequencies_cm-1": _to_builtin(frequencies_cm),
        "energies_meV": _to_builtin([energy * 1000.0 for energy in energies]),
        "frequency_unit": "cm-1",
        "energy_unit": "meV",
        "_ase_vib_energies_eV": energies,
    }


def _run_ideal_gas_thermochemistry(
    atoms,
    vib_energies,
    *,
    temperature: float,
    pressure: float,
    symmetry_number: int,
) -> dict:
    from ase.thermochemistry import IdealGasThermo

    potential_energy = float(atoms.get_potential_energy())
    if len(atoms) == 1:
        geometry = "monatomic"
    else:
        geometry = "linear" if _is_linear_atoms(atoms) else "nonlinear"

    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=potential_energy,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry_number,
        spin=0,
        ignore_imag_modes=True,
    )
    enthalpy = thermo.get_enthalpy(temperature=temperature, verbose=False)
    entropy = thermo.get_entropy(
        temperature=temperature,
        pressure=pressure,
        verbose=False,
    )
    gibbs = thermo.get_gibbs_energy(
        temperature=temperature,
        pressure=pressure,
        verbose=False,
    )
    return {
        "temperature_K": float(temperature),
        "pressure_Pa": float(pressure),
        "geometry": geometry,
        "symmetry_number": int(symmetry_number),
        "enthalpy_eV": float(enthalpy),
        "entropy_eV_per_K": float(entropy),
        "gibbs_free_energy_eV": float(gibbs),
    }


class _PySCFMoleculeCalculator(Calculator):
    """Small ASE calculator adapter backed by PySCF energies and gradients."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        spec: PySCFMoleculeSpec,
        *,
        device: PySCFDevice,
        max_cycle: int,
        conv_tol: float,
        chkfile: Optional[str],
    ):
        super().__init__()
        self.spec = spec
        self.device = device
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.chkfile = chkfile
        self.last_scf: dict[str, Any] = {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = self.atoms.get_positions().tolist()
        mol = _build_molecule(self.spec, positions=positions)
        mf = _build_molecule_scf(
            self.spec,
            mol,
            max_cycle=self.max_cycle,
            conv_tol=self.conv_tol,
            chkfile=self.chkfile,
        )
        mf = _apply_pyscf_device(mf, self.device, "molecular SCF")
        energy_hartree = float(mf.kernel())
        results: dict[str, Any] = {"energy": energy_hartree * HARTREE_TO_EV}
        if "forces" in properties:
            gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
            results["forces"] = -gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        self.last_scf = {
            "reference": self.spec.reference,
            "xc": self.spec.xc if self.spec.reference in {"RKS", "UKS"} else None,
            "converged": bool(getattr(mf, "converged", False)),
            "total_energy": _energy_payload(energy_hartree),
        }
        self.results = results


class _PySCFCrystalCalculator(Calculator):
    """ASE calculator adapter for PySCF periodic energies.

    Periodic forces are evaluated with central finite differences of PySCF PBC
    energies. This is intentionally simple and expensive, but it keeps the first
    crystal iteration dependency-light and explicit.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        spec: PySCFCrystalSpec,
        *,
        device: PySCFDevice,
        max_cycle: int,
        conv_tol: float,
        force_delta: float,
    ):
        super().__init__()
        self.spec = spec
        self.device = device
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.force_delta = force_delta
        self.last_scf: dict[str, Any] = {}

    def _energy_for_positions(self, positions) -> float:
        cell = _build_cell(self.spec, positions=np.asarray(positions).tolist())
        mf = _build_crystal_scf(
            self.spec,
            cell,
            max_cycle=self.max_cycle,
            conv_tol=self.conv_tol,
        )
        mf = _apply_pyscf_device(mf, self.device, "periodic SCF")
        energy_hartree = float(mf.kernel())
        self.last_scf = {
            "reference": self.spec.reference,
            "xc": self.spec.xc if "KS" in self.spec.reference else None,
            "converged": bool(getattr(mf, "converged", False)),
            "total_energy": _energy_payload(energy_hartree),
        }
        return energy_hartree * HARTREE_TO_EV

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        energy_ev = self._energy_for_positions(positions)
        results: dict[str, Any] = {"energy": energy_ev}
        if "forces" in properties:
            forces = np.zeros_like(positions)
            for atom_index in range(positions.shape[0]):
                for coord_index in range(positions.shape[1]):
                    plus = positions.copy()
                    minus = positions.copy()
                    plus[atom_index, coord_index] += self.force_delta
                    minus[atom_index, coord_index] -= self.force_delta
                    e_plus = self._energy_for_positions(plus)
                    e_minus = self._energy_for_positions(minus)
                    forces[atom_index, coord_index] = -(e_plus - e_minus) / (
                        2.0 * self.force_delta
                    )
            results["forces"] = forces
        self.results = results


def create_pyscf_molecule_core(
    structure_file: str,
    *,
    charge: int = 0,
    spin: int = 0,
    basis: str = "sto-3g",
    unit: PySCFUnit = "Angstrom",
    reference: PySCFMoleculeReference = "RHF",
    xc: Optional[str] = None,
    device: PySCFDevice = "cpu",
    fmt: Optional[str] = None,
    max_memory: int = 4000,
    verbose: int = 0,
    output_json: Optional[str] = None,
) -> dict:
    """Create a JSON-serializable PySCF molecule specification."""
    _require_pyscf()
    atoms, source_file = _read_ase_structure(structure_file, fmt=fmt)
    atom_payload = _ase_atoms_payload(atoms)
    spec = PySCFMoleculeSpec(
        source_structure_file=source_file,
        symbols=atom_payload["symbols"],
        positions=atom_payload["positions"],
        charge=charge,
        spin=spin,
        basis=basis,
        unit=unit,
        reference=reference,
        xc=xc,
        device=device,
        max_memory=max_memory,
        verbose=verbose,
        metadata={
            "formula": atom_payload["formula"],
            "source_format": fmt,
        },
    )

    mol = _build_molecule(spec)
    result = {
        "status": "success",
        "object_type": "pyscf_molecule",
        "pyscf_molecule": spec.model_dump(),
        "molecule": {
            "formula": atom_payload["formula"],
            "natoms": int(mol.natm),
            "nelectron": int(mol.nelectron),
            "charge": int(mol.charge),
            "spin": int(mol.spin),
            "basis": basis,
            "reference": reference,
            "xc": xc if reference in {"RKS", "UKS"} else None,
        },
        "artifacts": {},
    }
    _write_result(result, output_json)
    return result


def create_pyscf_crystal_core(
    structure_file: str,
    *,
    charge: int = 0,
    spin: int = 0,
    basis: str = "gth-szv",
    pseudo: Optional[str] = "gth-pade",
    unit: PySCFUnit = "Angstrom",
    reference: PySCFCrystalReference = "RKS",
    xc: Optional[str] = "pbe",
    kpts: Optional[list[int]] = None,
    device: PySCFDevice = "cpu",
    fmt: Optional[str] = None,
    max_memory: int = 4000,
    verbose: int = 0,
    output_json: Optional[str] = None,
) -> dict:
    """Create a JSON-serializable PySCF periodic Cell specification."""
    _require_pyscf()
    atoms, source_file = _read_ase_structure(structure_file, fmt=fmt)
    atom_payload = _ase_atoms_payload(atoms)
    lattice_vectors = atom_payload["cell"]
    if not lattice_vectors or not any(
        any(abs(float(x)) > 1e-12 for x in row) for row in lattice_vectors
    ):
        raise ValueError(
            "create_pyscf_crystal requires a structure file with nonzero "
            "lattice vectors."
        )

    spec = PySCFCrystalSpec(
        source_structure_file=source_file,
        symbols=atom_payload["symbols"],
        positions=atom_payload["positions"],
        lattice_vectors=lattice_vectors,
        pbc=atom_payload["pbc"],
        charge=charge,
        spin=spin,
        basis=basis,
        pseudo=pseudo,
        unit=unit,
        reference=reference,
        xc=xc,
        kpts=kpts or [1, 1, 1],
        device=device,
        max_memory=max_memory,
        verbose=verbose,
        metadata={
            "formula": atom_payload["formula"],
            "source_format": fmt,
        },
    )

    cell = _build_cell(spec)
    result = {
        "status": "success",
        "object_type": "pyscf_crystal",
        "pyscf_crystal": spec.model_dump(),
        "crystal": {
            "formula": atom_payload["formula"],
            "natoms": int(cell.natm),
            "nelectron": int(cell.nelectron),
            "charge": int(cell.charge),
            "spin": int(cell.spin),
            "basis": basis,
            "pseudo": pseudo,
            "reference": reference,
            "xc": xc if "KS" in reference else None,
            "kpts": spec.kpts,
        },
        "artifacts": {},
    }
    _write_result(result, output_json)
    return result


def run_pyscf_molecule_core(
    pyscf_molecule: Mapping[str, Any] | PySCFMoleculeSpec | None = None,
    *,
    pyscf_molecule_json: Optional[str] = None,
    driver: PySCFDriver = "optimization",
    device: Optional[PySCFDevice] = None,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    steps: int = 100,
    displacement: float = 0.01,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    symmetry_number: int = 1,
    optimize_before_analysis: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-9,
    chkfile: Optional[str] = None,
    output_json: Optional[str] = "pyscf_molecule_results.json",
) -> dict:
    """Run a PySCF-backed molecular workflow."""
    started_at = time.time()
    pyscf = _require_pyscf()
    spec, molecule_json_path = _resolve_molecule_spec(
        pyscf_molecule, pyscf_molecule_json
    )
    run_device = device or spec.device
    driver_name = _normalise_driver(driver)

    atoms = _atoms_from_molecule_spec(spec)
    calculator = _PySCFMoleculeCalculator(
        spec,
        device=run_device,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        chkfile=chkfile,
    )
    atoms.calc = calculator

    optimization: dict[str, Any] = {}
    vibrations: dict[str, Any] = {}
    thermochemistry: dict[str, Any] = {}

    if driver_name in {"optimization", "vibration", "thermochemistry"}:
        if driver_name == "optimization" or optimize_before_analysis:
            converged = _run_optimizer(atoms, optimizer, fmax, steps)
            optimization = {
                "optimizer": optimizer,
                "converged": converged,
                "fmax_eV_per_Angstrom": float(fmax),
                "steps": int(steps),
            }

    energy_ev = float(atoms.get_potential_energy())

    if driver_name in {"vibration", "thermochemistry"}:
        vibrations = _run_vibrations(atoms, displacement)

    if driver_name == "thermochemistry":
        thermochemistry = _run_ideal_gas_thermochemistry(
            atoms,
            vibrations.pop("_ase_vib_energies_eV"),
            temperature=temperature,
            pressure=pressure,
            symmetry_number=symmetry_number,
        )
    else:
        vibrations.pop("_ase_vib_energies_eV", None)

    result = {
        "status": "success",
        "calculation": "pyscf_molecule",
        "driver": driver_name,
        "device": run_device,
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "input": {
            "pyscf_molecule": spec.model_dump(),
            "pyscf_molecule_json": molecule_json_path,
            "max_cycle": max_cycle,
            "conv_tol": conv_tol,
        },
        "scf": calculator.last_scf,
        "energy": _energy_payload_from_ev(energy_ev),
        "final_structure": _ase_atoms_payload(atoms),
        "optimization": optimization,
        "vibrations": vibrations,
        "thermochemistry": thermochemistry,
        "artifacts": {
            "chkfile": str(Path(chkfile).resolve()) if chkfile else None,
        },
        "wall_time": time.time() - started_at,
    }
    _write_result(result, output_json)
    return result


def run_pyscf_crystal_core(
    pyscf_crystal: Mapping[str, Any] | PySCFCrystalSpec | None = None,
    *,
    pyscf_crystal_json: Optional[str] = None,
    driver: PySCFDriver = "energy",
    device: Optional[PySCFDevice] = None,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    steps: int = 50,
    displacement: float = 0.01,
    force_delta: float = 0.005,
    optimize_before_analysis: bool = False,
    max_cycle: int = 50,
    conv_tol: float = 1e-9,
    output_json: Optional[str] = "pyscf_crystal_results.json",
) -> dict:
    """Run a PySCF-backed periodic workflow."""
    started_at = time.time()
    pyscf = _require_pyscf()
    spec, crystal_json_path = _resolve_crystal_spec(pyscf_crystal, pyscf_crystal_json)
    run_device = device or spec.device
    driver_name = _normalise_driver(driver)

    if driver_name == "thermochemistry":
        result = {
            "status": "failure",
            "calculation": "pyscf_crystal",
            "driver": driver_name,
            "error_type": "NotImplementedError",
            "message": (
                "Crystal thermochemistry requires a phonon density-of-states "
                "workflow, which is not implemented in the first PySCF MCP iteration."
            ),
            "artifacts": {},
            "wall_time": time.time() - started_at,
        }
        _write_result(result, output_json)
        return result

    atoms = _atoms_from_crystal_spec(spec)
    calculator = _PySCFCrystalCalculator(
        spec,
        device=run_device,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        force_delta=force_delta,
    )
    atoms.calc = calculator

    optimization: dict[str, Any] = {}
    vibrations: dict[str, Any] = {}
    if driver_name in {"optimization", "vibration"}:
        if driver_name == "optimization" or optimize_before_analysis:
            converged = _run_optimizer(atoms, optimizer, fmax, steps)
            optimization = {
                "optimizer": optimizer,
                "converged": converged,
                "fmax_eV_per_Angstrom": float(fmax),
                "steps": int(steps),
                "force_method": "finite_difference_energy",
                "force_delta_Angstrom": float(force_delta),
                "cell_relaxation": "fixed_cell",
            }

    energy_ev = float(atoms.get_potential_energy())

    if driver_name == "vibration":
        vibrations = _run_vibrations(atoms, displacement)
        vibrations.pop("_ase_vib_energies_eV", None)
        vibrations["scope"] = "Gamma-point finite differences with fixed cell"

    result = {
        "status": "success",
        "calculation": "pyscf_crystal",
        "driver": driver_name,
        "device": run_device,
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "input": {
            "pyscf_crystal": spec.model_dump(),
            "pyscf_crystal_json": crystal_json_path,
            "max_cycle": max_cycle,
            "conv_tol": conv_tol,
        },
        "scf": calculator.last_scf,
        "energy": _energy_payload_from_ev(energy_ev),
        "final_structure": _ase_atoms_payload(atoms),
        "optimization": optimization,
        "vibrations": vibrations,
        "artifacts": {},
        "wall_time": time.time() - started_at,
        "notes": [
            "Crystal optimization and vibration currently use finite-difference "
            "forces from PySCF periodic energies with fixed lattice vectors."
        ],
    }
    _write_result(result, output_json)
    return result
