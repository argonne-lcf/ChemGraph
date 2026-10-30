"""Strict ORCA 6.x single-ground-state RKS output parser.

Coverage is fixture-based. Unsupported or incomplete evidence fails explicitly.
"""

import hashlib
from pathlib import Path
import re

import numpy as np
from ase import __version__ as ase_version
from ase.data import atomic_numbers
from ase.units import Hartree

from chemgraph.schemas.electronic import ElectronicStructureResult

NUMBER = r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?"


class OrcaParseError(ValueError):
    def __init__(self, reason, detail):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def fail(reason, detail):
    raise OrcaParseError(reason, detail)


def section(text, title):
    match = re.search(r"(?m)^" + re.escape(title) + r"\s*\n-+\s*\n", text)
    if not match:
        return ""
    tail = text[match.end() :]
    end = re.search(r"(?m)^[-=]{5,}\s*\n[^\n]+\n[-=]{5,}\s*$", tail)
    return tail[: end.start()] if end else tail


def native_settings(text):
    titles = (
        "SCF SETTINGS",
        "BASIS SET INFORMATION",
        "AUXILIARY/J BASIS SET INFORMATION",
        "DFT GRID GENERATION",
        "COSX GRID GENERATION",
    )
    return {title: section(text, title) or "not_reported" for title in titles}


def _one(text, pattern, label):
    matches = re.findall(pattern, text, flags=re.M | re.I)
    if len(matches) != 1:
        fail("missing_or_ambiguous_evidence", label)
    return matches[0]


def validate_native(
    text, *, charge, multiplicity, symbols, positions, allow_gradient=False
):
    """Validate one native SCF result before using either energy or forces."""
    version = _one(text, r"Program Version\s+(\d+\.\d+\.\d+)", "engine version")
    if not version.startswith("6."):
        fail("unsupported_version", version)
    echoed = "\n".join(re.findall(r"^\s*\|\s*\d+> ?(.*)$", text, re.M))
    if not echoed:
        fail("unsupported_native_task", "Missing echoed input")
    active = "\n".join(line.split('#')[0] for line in echoed.splitlines()).lower()
    blocks = re.findall(r"%([a-z_]+)", active)
    if any(b not in {"output", "scf", "pal"} for b in blocks):
        fail("unsupported_native_task", "Only one ground-state RKS task is supported")
    keywords = " ".join(re.findall(r"^\s*!([^\n]*)", active, re.M)).split()
    # Only the documented gas-phase ground-state input family is accepted.
    # Unknown native keywords may request solvent, excited-state/property work,
    # ECPs, or override the advertised method; do not silently relabel those.
    supported = {
        "rks",
        "pbe",
        "pbe0",
        "b3lyp",
        "bp86",
        "sto-3g",
        "def2-svp",
        "def2-tzvp",
        "def2-tzvpd",
        "sp",
        "tightscf",
        "normalscf",
        "verytightscf",
        "noautostart",
        "usesym",
        "nosym",
    }
    if allow_gradient:
        supported.add("engrad")
    if (
        set(keywords) - supported
        or '$new_job' in active
        or '$compound' in active
        or re.search(
            r"(?m)^\s*(?:ORCA (?:NMR|TD-DFT|CASSCF)|GEOMETRY OPTIMIZATION|VIBRATIONAL FREQUENCIES)",
            text,
        )
    ):
        fail("unsupported_native_task", "Multiple/unsupported native task")
    if (
        len(re.findall(r"FINAL SINGLE POINT ENERGY", text)) != 1
        or len(re.findall(r"SCF CONVERGED AFTER", text)) > 1
        or len(re.findall(r"(?m)^ORBITAL ENERGIES\s*$", text)) > 1
        or len(re.findall(r"(?m)^CARTESIAN COORDINATES \(ANGSTROEM\)\s*$", text)) != 1
    ):
        fail("unsupported_native_task", "Expected exactly one SCF task")
    if re.search(
        r"SCF (?:NOT CONVERGED|DID NOT CONVERGE)|SCF CONVERGENCE FAILURE|ORCA finished by error|aborting the run|SCF IS NOT CONVERGED",
        text,
        re.I,
    ):
        fail("scf_failure", "Contradictory native failure marker")
    cycles = int(
        _one(text, r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", "native SCF convergence")
    )
    if text.count("ORCA TERMINATED NORMALLY") != 1:
        fail("abnormal_termination", "Normal termination is required")
    if not re.search(r"Density Functional\s+Method\s+\.+\s+DFT", text) or not re.search(
        r"HFTyp\s+\.+\s+RHF\b", text
    ):
        fail("unsupported_spin", "A restricted DFT result is required")
    actual_charge = int(_one(text, r"Total Charge\s+Charge\s+\.+\s+(-?\d+)", "charge"))
    mult = int(_one(text, r"Multiplicity\s+Mult\s+\.+\s+(\d+)", "multiplicity"))
    nel = int(_one(text, r"Number of Electrons\s+NEL\s+\.+\s+(\d+)", "electron count"))
    if (
        actual_charge != charge
        or mult != multiplicity
        or mult != 1
        or nel != sum(atomic_numbers[s] for s in symbols) - charge
        or nel % 2
    ):
        fail("identity_mismatch", "Charge, spin or electron count disagrees with input")
    geometry = section(text, "CARTESIAN COORDINATES (ANGSTROEM)")
    rows = re.findall(
        r"^\s*([A-Z][a-z]?)\s+("
        + NUMBER
        + r")\s+("
        + NUMBER
        + r")\s+("
        + NUMBER
        + r")\s*$",
        geometry,
        re.M,
    )
    if [r[0] for r in rows] != list(symbols) or not np.allclose(
        np.array([r[1:] for r in rows], float), positions, atol=5.1e-7, rtol=0
    ):
        fail("geometry_mismatch", "Native geometry/atom order differs from invocation")
    energy = (
        float(_one(text, r"FINAL SINGLE POINT ENERGY\s+(" + NUMBER + r")", "energy"))
        * Hartree
    )
    if not np.isfinite(energy):
        fail("invalid_energy", "Nonfinite energy")
    return dict(
        version=version,
        cycles=cycles,
        electron_count=nel,
        energy=energy,
        echoed_input=echoed,
        native_settings=native_settings(text),
    )


def resolution(token):
    from decimal import Decimal

    return 10.0 ** Decimal(token).as_tuple().exponent


def _energy_table(text):
    body = section(text, "ORBITAL ENERGIES")
    rows = []
    for line in body.splitlines():
        if re.match(r"\s*\d+\s+", line):
            bits = line.split()
            if len(bits) < 4:
                fail("invalid_spectrum", "Malformed orbital row")
            try:
                idx, occ, eh, ev = (
                    int(bits[0]),
                    float(bits[1]),
                    float(bits[2]),
                    float(bits[3]),
                )
                precision = resolution(bits[2])
                # ORCA manual, Units and Conversion Factors: printed eV uses
                # 27.2113834; stored energies below use ASE's Hartree constant.
                if (
                    abs(eh * 27.2113834 - ev)
                    > (precision * 27.2113834 + resolution(bits[3])) / 2 + 1e-7
                ):
                    fail("inconsistent_spectrum", "Hartree/eV columns disagree")
                rows.append((idx, occ, eh, precision))
            except (ValueError, IndexError) as exc:
                fail("invalid_spectrum", str(exc))
    return rows


def _mo_headers(text):
    lines = section(text, "MOLECULAR ORBITALS (RHF, ROHF)").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if re.fullmatch(r"\s*\d+(?:\s+\d+)*\s*", line):
            try:
                indices = [int(s) for s in line.split()]
                energies = lines[i + 1].split()
                occupations = lines[i + 2].split()
                if len(indices) != len(energies) or len(indices) != len(occupations):
                    fail("invalid_spectrum", "MO header length mismatch")
                rows.extend(
                    (idx, float(occ), float(eh), resolution(eh))
                    for idx, occ, eh in zip(indices, occupations, energies)
                )
            except (IndexError, ValueError) as exc:
                fail("invalid_spectrum", str(exc))
    return rows


def parse_orca_output(
    text,
    *,
    charge,
    multiplicity,
    symbols,
    positions,
    method,
    basis,
    input_path,
    output_path,
    convergence_policy="native",
    molecule_id=None,
    conformer_id=None,
):
    evidence = validate_native(
        text,
        charge=charge,
        multiplicity=multiplicity,
        symbols=symbols,
        positions=positions,
    )

    def normalized_input(value):
        return [
            " ".join(line.split())
            for line in value.splitlines()
            if line.strip() and "****END OF INPUT****" not in line
        ]

    if normalized_input(Path(input_path).read_text()) != normalized_input(
        evidence['echoed_input']
    ):
        fail("provenance_mismatch", "Native input artifact disagrees with echoed input")
    keywords = (
        ' '.join(re.findall(r"^\s*!([^\n]*)", evidence['echoed_input'], re.M))
        .lower()
        .split()
    )
    if method.lower() not in keywords or basis.lower() not in keywords:
        fail("settings_mismatch", "Requested method/basis absent from echoed input")
    nbf_values = re.findall(r"Number of basis functions\s+\.+\s+(\d+)", text)
    if not nbf_values or len(set(nbf_values)) != 1:
        fail("unsupported_basis", "Missing/ambiguous basis dimension")
    nbf = int(nbf_values[0])
    reductions = re.findall(
        r"Number of eigenvalues below threshold\s+\.+\s+(\d+)", text
    )
    if reductions != ['0'] or re.search(
        r"(?:remov(?:ed|ing)|discard(?:ed|ing))\s+[1-9]\d*\s+(?:basis|orbital|eigen)",
        text,
        re.I,
    ):
        fail(
            "unsupported_basis_reduction",
            "Require explicit zero eigenvalues below threshold",
        )
    table, headers = _energy_table(text), _mo_headers(text)
    for rows in (table, headers):
        if rows and (
            [r[0] for r in rows] != list(range(len(rows)))
            or len(rows) > nbf
            or not np.isfinite(np.array(rows)).all()
        ):
            fail("invalid_spectrum", "Duplicate/missing indices or nonfinite values")
    for a, b in zip(table, headers):
        if abs(a[1] - b[1]) > 1e-6 or abs(a[2] - b[2]) > (a[3] + b[3]) / 2 + 1e-10:
            fail("inconsistent_spectrum", "Energy table and MO headers disagree")
    rows, source = (
        (table, 'orbital_energy_table') if len(table) == nbf else (headers, 'mo_header')
    )
    if len(rows) != nbf:
        fail("incomplete_spectrum", f"Require all {nbf} orbitals")
    orbitals = [dict(index=i, occupation=o, energy=e * Hartree) for i, o, e, _ in rows]
    occupied, virtual = (
        [r for r in orbitals if r['occupation'] > 1],
        [r for r in orbitals if r['occupation'] < 1],
    )
    if not occupied or not virtual:
        fail("invalid_spectrum", "Missing occupied/virtual orbitals")
    homo, lumo = max(r['energy'] for r in occupied), min(r['energy'] for r in virtual)
    precision = max(r[3] for r in rows) * Hartree
    artifacts = {}
    for kind, path in (("input", input_path), ("output", output_path)):
        path = Path(path).resolve()
        artifacts[kind] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if Path(output_path).read_text() != text:
        fail("provenance_mismatch", "Parsed text differs from native output artifact")
    return ElectronicStructureResult(
        total_energy=evidence['energy'],
        charge=charge,
        electron_count=evidence['electron_count'],
        method=method,
        basis=basis,
        nbf=nbf,
        nmo=nbf,
        orbitals=orbitals,
        homo=homo,
        lumo=lumo,
        gap=lumo - homo,
        homo_indices=[
            r['index']
            for r in occupied
            if abs(r['energy'] - homo) <= max(1e-4, precision)
        ],
        lumo_indices=[
            r['index']
            for r in virtual
            if abs(r['energy'] - lumo) <= max(1e-4, precision)
        ],
        source=source,
        energy_resolution_ev=precision,
        scf_evidence={
            "native_result": "SCF CONVERGED AFTER",
            "cycles": evidence['cycles'],
        },
        convergence_policy=convergence_policy,
        engine="ORCA",
        engine_version=evidence['version'],
        ase_version=ase_version,
        artifacts=artifacts,
        working_directory=str(Path(output_path).resolve().parent),
        native_settings=evidence['native_settings'],
        molecule_id=molecule_id,
        conformer_id=conformer_id,
        warnings=[
            "Anion calculation lacks a diffuse basis; electron binding is not established"
        ]
        if charge < 0 and basis.lower() != 'def2-tzvpd'
        else [],
    )
