"""Explicitly synthetic ORCA output for offline execution tests, never live evidence."""

from ase.units import Hartree


def synthetic_output(
    atoms,
    *,
    charge=0,
    functional='PBE',
    basis='def2-SVP',
    energy=-40.0,
    gradient=False,
    nbf=None,
    homo_positive=False,
):
    nel = sum(atoms.numbers) - charge
    occupied = int(nel // 2)
    nbf = nbf or occupied + 25
    lines = [
        "Program Version 6.0.0",
        f"| 1> ! RKS {functional} {basis} {'EnGrad' if gradient else 'SP'}",
        "CARTESIAN COORDINATES (ANGSTROEM)",
        "---------------------------------",
    ]
    lines += [
        f"{s} {x:.6f} {y:.6f} {z:.6f}"
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)
    ]
    lines += [
        "----------------",
        "SCF SETTINGS",
        "----------------",
        "Density Functional Method .... DFT(GTOs)",
        "Hartree-Fock type HFTyp .... RHF",
        f"Total Charge Charge .... {charge}",
        "Multiplicity Mult .... 1",
        f"Number of Electrons NEL .... {nel}",
        f"Number of basis functions ... {nbf}",
        "Number of eigenvalues below threshold ... 0",
        "SCF CONVERGED AFTER 6 CYCLES",
        "----------------",
        "ORBITAL ENERGIES",
        "----------------",
        "NO OCC E(Eh) E(eV)",
    ]
    for i in range(nbf):
        eh = -0.5 + i * 0.5 / occupied if i < occupied else 0.1 + (i - occupied) * 0.01
        if homo_positive and i < occupied:
            eh += 1
        lines.append(
            f"{i} {2 if i < occupied else 0:.4f} {eh:.8f} {eh * 27.2113834:.6f}"
        )
    lines += ["----------------", "END OF SPECTRUM", "----------------"]
    if gradient:
        lines.append("ORCA SCF GRADIENT CALCULATION")
    lines += [
        f"FINAL SINGLE POINT ENERGY {energy / Hartree:.12f}",
        "ORCA TERMINATED NORMALLY",
    ]
    return '\n'.join(lines) + '\n'


def synthetic_engrad(atoms, energy=-40.0, force=0.0):

    rows = [
        '# Number of atoms',
        str(len(atoms)),
        '# The current total energy',
        f'{energy / Hartree:.12f}',
        '# The current gradient',
    ]
    rows += [str(force) for _ in range(3 * len(atoms))]
    rows += ['# The atomic numbers and current coordinates in Bohr']
    rows += [
        ' '.join(map(str, [n, *(xyz / 0.5291772083)]))
        for n, xyz in zip(atoms.numbers, atoms.positions)
    ]
    return '\n'.join(rows) + '\n'
