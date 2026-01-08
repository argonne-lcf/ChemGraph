### Helper functions for mcp_tools.py ###

from chemgraph.models.atomsdata import AtomsData


def load_calculator(calculator: dict) -> tuple[object, dict, dict]:
    """Load an ASE calculator based on the provided configuration.

    Parameters
    ----------
    calculator : dict
        Dictionary containing calculator configuration parameters

    Returns
    -------
    object
        ASE calculator instance

    Raises
    ------
    ValueError
        If the calculator type is not supported
    """
    calc_type = calculator["calculator_type"].lower()

    if "emt" in calc_type:
        from chemgraph.models.calculators.emt_calc import EMTCalc

        calc = EMTCalc(**calculator)
    elif "tblite" in calc_type:
        from chemgraph.models.calculators.tblite_calc import TBLiteCalc

        calc = TBLiteCalc(**calculator)
    elif "orca" in calc_type:
        from chemgraph.models.calculators.orca_calc import OrcaCalc

        calc = OrcaCalc(**calculator)

    elif "nwchem" in calc_type:
        from chemgraph.models.calculators.nwchem_calc import NWChemCalc

        calc = NWChemCalc(**calculator)

    elif "fairchem" in calc_type:
        from chemgraph.models.calculators.fairchem_calc import FAIRChemCalc

        calc = FAIRChemCalc(**calculator)

    elif "mace" in calc_type:
        from chemgraph.models.calculators.mace_calc import MaceCalc

        calc = MaceCalc(**calculator)

    elif "aimnet2" in calc_type:
        from chemgraph.models.calculators.aimnet2_calc import AIMNET2Calc

        calc = AIMNET2Calc(**calculator)

    else:
        raise ValueError(
            f"Unsupported calculator: {calculator}. Available calculators are EMT, TBLite (GFN2-xTB, GFN1-xTB), Orca and FAIRChem or MACE or AIMNET2."
        )
    # Extract additional args like spin/charge if the model defines it
    extra_info = {}
    if hasattr(calc, "get_atoms_properties"):
        extra_info = calc.get_atoms_properties()

    return calc.get_calculator(), extra_info, calc


def atoms_to_atomsdata(atoms):
    """Convert ASE Atoms object to AtomsData.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object

    Returns
    -------
    AtomsData
        ChemGraph AtomsData object
    """
    return AtomsData(
        numbers=atoms.numbers.tolist(),
        positions=atoms.positions.tolist(),
        cell=atoms.cell.tolist(),
        pbc=atoms.pbc.tolist(),
    )


def is_linear_molecule(atomsdata: AtomsData, tol=1e-3) -> bool:
    """Determine if a molecule is linear or not.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object containing the molecular structure
    tol : float, optional
        Tolerance to check for linear molecule, by default 1e-3

    Returns
    -------
    bool
        True if the molecule is linear, False otherwise
    """
    import numpy as np

    coords = np.array(atomsdata.positions)
    # Center the coordinates.
    centered = coords - np.mean(coords, axis=0)
    # Singular value decomposition.
    U, s, Vt = np.linalg.svd(centered)
    # For a linear molecule, only one singular value is significantly nonzero.
    if s[0] == 0:
        return False  # degenerate case (all atoms at one point)
    return (s[1] / s[0]) < tol


def get_symmetry_number(atomsdata: AtomsData) -> int:
    """Get the rotational symmetry number of a molecule using Pymatgen.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object containing the molecular structure

    Returns
    -------
    int
        Rotational symmetry number of the molecule
    """
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = Atoms(
        numbers=atomsdata.numbers,
        positions=atomsdata.positions,
        cell=atomsdata.cell,
        pbc=atomsdata.pbc,
    )

    aaa = AseAtomsAdaptor()
    molecule = aaa.get_molecule(atoms)
    pga = PointGroupAnalyzer(molecule)
    symmetrynumber = pga.get_rotational_symmetry_number()

    return symmetrynumber
