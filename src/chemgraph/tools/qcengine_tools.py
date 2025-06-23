from chemgraph.state.state import MultiAgentState
from langchain_core.messages import HumanMessage
import json
import qcengine
import qcelemental as qcel
import numpy as np
from chemgraph.utils.logging_config import setup_logger
from chemgraph.models.atomsdata import AtomsData
from ase.data import atomic_numbers

logger = setup_logger(__name__)


def run_qcengine(state: MultiAgentState, program="psi4"):
    """Run a QCEngine calculation.

    This function executes a quantum chemistry calculation using QCEngine
    with the specified program and parameters from the state.

    Parameters
    ----------
    state : MultiAgentState
        The state of the multi-agent system containing calculation parameters
    program : str, optional
        The quantum chemistry program to use, by default "psi4"

    Returns
    -------
    dict
        Dictionary containing the calculation results in a format suitable for
        the multi-agent system

    Raises
    ------
    Exception
        If the calculation fails or encounters an error
    """
    params = state["parameter_response"][-1]
    input = json.loads(params.content)
    program = input["program"]

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert NumPy array to list
            if isinstance(obj, np.generic):
                return obj.item()  # Convert NumPy scalar to Python scalar
            return super().default(obj)

    def parse_atomsdata_to_mol(input):
        numbers = input["atomsdata"]["numbers"]
        positions = input["atomsdata"]["positions"]

        # Convert atomic numbers to element symbols
        symbols = [atomic_numbers[num] for num in numbers]

        # Flatten positions list for QCEngine format
        geometry = [coord for position in positions for coord in position]
        return {"symbols": symbols, "geometry": geometry}

    input["molecule"] = parse_atomsdata_to_mol(input)
    del input["atomsdata"]
    del input["program"]
    try:
        logger.info("Starting QCEngine calculation")
        result = qcengine.compute(input, program).dict()
        del result["stdout"]
        output = []
        output.append(HumanMessage(role="system", content=json.dumps(result, cls=NumpyEncoder)))
        logger.info("QCEngine calculation completed successfully")
        return {"opt_response": output}
    except Exception as e:
        logger.error(f"Error in QCEngine calculation: {str(e)}")
        raise


def is_linear_molecule(coords, tol=1e-3):
    """Determine if a molecule is linear.

    This function uses singular value decomposition to determine if a molecule
    is linear based on its atomic coordinates.

    Parameters
    ----------
    coords : np.ndarray
        (N x 3) array of Cartesian coordinates.
    tol : float, optional
        Tolerance for linearity; if the ratio of the second largest to largest
        singular value is below tol, the molecule is considered linear,
        by default 1e-3

    Returns
    -------
    bool
        True if the molecule is linear, False otherwise

    Notes
    -----
    A molecule is considered linear if the ratio of the second largest to largest
    singular value of its centered coordinates is below the tolerance.
    """
    # Center the coordinates.
    centered = coords - np.mean(coords, axis=0)
    # Singular value decomposition.
    U, s, Vt = np.linalg.svd(centered)
    # For a linear molecule, only one singular value is significantly nonzero.
    if s[0] == 0:
        return False  # degenerate case (all atoms at one point)
    return (s[1] / s[0]) < tol


def compute_mass_weighted_hessian(hessian, masses):
    """Mass-weights the Hessian matrix.

    This function converts a Hessian matrix to its mass-weighted form,
    which is required for vibrational frequency calculations.

    Parameters
    ----------
    hessian : np.ndarray
        A (3N x 3N) Hessian matrix in atomic units (e.g. Hartree/Bohr²).
    masses : array-like
        A list/array of atomic masses in amu.

    Returns
    -------
    F : np.ndarray
        The mass-weighted Hessian.
    mass_vector : np.ndarray
        The mass vector (each mass repeated three times).

    Notes
    -----
    The mass-weighting is performed by multiplying the Hessian by the inverse
    square root of the mass matrix. This is a standard step in normal mode
    analysis.
    """
    # Convert masses from amu to atomic units (electron masses)
    amu_to_au = 1822.888486
    masses_au = np.array(masses) * amu_to_au
    # Each atom has 3 coordinates.
    mass_vector = np.repeat(masses_au, 3)
    inv_sqrt_m = 1.0 / np.sqrt(mass_vector)
    M_inv_sqrt = np.diag(inv_sqrt_m)
    # Mass-weighted Hessian.
    F = M_inv_sqrt @ hessian @ M_inv_sqrt
    return F, mass_vector


def build_projection_operator(masses, coords, is_linear=False):
    """Build a projection operator for translational and rotational modes.

    This function constructs a projection operator that removes translational
    and rotational degrees of freedom from the Hessian matrix.

    Parameters
    ----------
    masses : array-like
        List of atomic masses (in amu) for N atoms.
    coords : np.ndarray
        (N x 3) array of Cartesian coordinates.
    is_linear : bool, optional
        If True, the molecule is assumed linear (removes 3 translations and 2 rotations);
        otherwise, it removes 3 translations and 3 rotations, by default False

    Returns
    -------
    P : np.ndarray
        A (3N x 3N) projection matrix.

    Notes
    -----
    The projection operator is used to remove the translational and rotational
    modes from the Hessian matrix before computing vibrational frequencies.
    For linear molecules, only 2 rotational modes are removed instead of 3.
    """
    N = len(masses)
    dim = 3 * N

    # Build translational modes: each column corresponds to translation in x, y, or z.
    trans_modes = np.zeros((dim, 3))
    for i in range(N):
        trans_modes[3 * i : 3 * i + 3, :] = np.eye(3)

    # Center of mass.
    masses_arr = np.array(masses)
    com = np.sum(coords * masses_arr[:, None], axis=0) / np.sum(masses_arr)
    disp = coords - com

    if not is_linear:
        # For non-linear molecules, build three rotational modes.
        rot_modes = np.zeros((dim, 3))
        for i in range(N):
            x, y, z = disp[i]
            # Rotation about x-axis: (0, -z, y)
            rot_modes[3 * i : 3 * i + 3, 0] = [0, -z, y]
            # Rotation about y-axis: (z, 0, -x)
            rot_modes[3 * i : 3 * i + 3, 1] = [z, 0, -x]
            # Rotation about z-axis: (-y, x, 0)
            rot_modes[3 * i : 3 * i + 3, 2] = [-y, x, 0]
        modes = np.hstack((trans_modes, rot_modes))  # shape: (dim, 6)
    else:
        # For linear molecules, only 2 independent rotations exist.
        # Determine the molecular (principal) axis.
        # For diatomics, this is simply the normalized vector between the two atoms.
        if N == 2:
            axis = coords[1] - coords[0]
        else:
            # Use SVD on centered coordinates.
            U, s, Vt = np.linalg.svd(disp)
            axis = Vt[0]  # principal axis (largest singular value)
        axis = axis / np.linalg.norm(axis)
        # Choose an arbitrary vector not parallel to 'axis' to form the first perpendicular direction.
        arbitrary = np.array([1, 0, 0])
        if np.allclose(np.abs(np.dot(arbitrary, axis)), 1.0, atol=1e-3):
            arbitrary = np.array([0, 1, 0])
        perp1 = np.cross(axis, arbitrary)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        perp2 /= np.linalg.norm(perp2)

        # Build rotational modes corresponding to rotations about perp1 and perp2.
        rot_modes = np.zeros((dim, 2))
        for i in range(N):
            # For rotation about perp1, displacement is: perp1 x (r_i - COM)
            rot1 = np.cross(perp1, disp[i])
            # For rotation about perp2, displacement is: perp2 x (r_i - COM)
            rot2 = np.cross(perp2, disp[i])
            rot_modes[3 * i : 3 * i + 3, 0] = rot1
            rot_modes[3 * i : 3 * i + 3, 1] = rot2
        modes = np.hstack((trans_modes, rot_modes))  # shape: (dim, 5)

    # Orthonormalize the modes using QR decomposition.
    Q, _ = np.linalg.qr(modes)
    # The projection operator that removes these modes.
    P = np.eye(dim) - Q @ Q.T
    return P


def compute_vibrational_frequencies(hessian, masses, coords=None, linear=None):
    """Calculate vibrational frequencies from a Hessian matrix.

    This function computes the vibrational frequencies of a molecule from its
    Hessian matrix and atomic masses. It can optionally project out translational
    and rotational modes if coordinates are provided.

    Parameters
    ----------
    hessian : np.ndarray
        A (3N x 3N) Hessian matrix in atomic units.
    masses : array-like
        List/array of atomic masses in amu.
    coords : np.ndarray or None, optional
        (N x 3) Cartesian coordinates. If provided, used to project out rotation
        and translation, by default None
    linear : bool or None, optional
        If set, explicitly specifies whether the molecule is linear.
        If None and coords is provided, linearity is determined automatically,
        by default None

    Returns
    -------
    frequencies_cm : np.ndarray
        Array of vibrational frequencies in cm⁻¹ (negative values indicate
        imaginary modes).

    Notes
    -----
    For non-linear molecules, 3 translational and 3 rotational modes are removed.
    For linear molecules (including diatomics), 3 translational and 2 rotational
    modes are removed.
    """
    F, _ = compute_mass_weighted_hessian(hessian, masses)

    # If coordinates are provided, project out translation and rotation.
    if coords is not None:
        if linear is None:
            # Automatically determine linearity.
            linear = is_linear_molecule(coords)
        P = build_projection_operator(masses, coords, is_linear=linear)
        F = P @ F @ P

    # Diagonalize the (projected) mass-weighted Hessian.
    eigenvals, _ = np.linalg.eigh(F)

    # Conversion factor from atomic unit of angular frequency to cm⁻¹.
    conv_factor = 219474.63  # approximate conversion constant

    frequencies_cm = []
    for lam in eigenvals:
        # For negative eigenvalues (imaginary frequencies), take negative of the converted value.
        if lam < 0:
            freq = -conv_factor * np.sqrt(abs(lam))
        else:
            freq = conv_factor * np.sqrt(lam)
        frequencies_cm.append(freq)

    frequencies_cm = np.array(frequencies_cm)

    # Filter out modes close to zero (the removed translations and rotations).
    threshold = 1e-3
    vib_frequencies = frequencies_cm[np.abs(frequencies_cm) > threshold]
    vib_frequencies = np.sort(vib_frequencies)
    return vib_frequencies


def convert_atomsdata_to_qcmolecule(atomsdata: AtomsData) -> qcel.models.Molecule:
    """Convert an AtomsData object to a QCElemental Molecule.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object containing the molecular structure

    Returns
    -------
    qcel.models.Molecule
        QCElemental Molecule object

    Notes
    -----
    This function converts atomic numbers to element symbols and
    converts coordinates from Angstroms to Bohr units.
    """
    import numpy as np

    atomic_numbers_dict = {value: key for key, value in atomic_numbers.items()}
    numbers = atomsdata.numbers
    symbols = [atomic_numbers_dict[num] for num in numbers]

    # atomsdata positions are in Angstrom. Convert to atomic unit for qcelemental.
    positions = np.array(atomsdata.positions)

    geometry = np.array([
        position * qcel.constants.conversion_factor("angstrom", "bohr") for position in positions
    ])
    molecule = qcel.models.Molecule(symbols=symbols, geometry=geometry.flatten())
    return molecule


def convert_qcmolecule_to_atomsdata(molecule: qcel.models.Molecule) -> AtomsData:
    """Convert a QCElemental Molecule to an AtomsData object.

    Parameters
    ----------
    molecule : qcel.models.Molecule
        QCElemental Molecule object

    Returns
    -------
    AtomsData
        AtomsData object

    Notes
    -----
    This function converts element symbols to atomic numbers and
    converts coordinates from Bohr to Angstrom units.
    """
    symbols = molecule.symbols
    geometry = molecule.geometry * qcel.constants.conversion_factor("bohr", "angstrom")
    positions = geometry.reshape(-1, 3)
    numbers = [atomic_numbers[sym] for sym in symbols]

    atomsdata = AtomsData(numbers=numbers, positions=positions)

    return atomsdata


def run_qcengine_multi_framework(state: MultiAgentState):
    """Run a QCEngine calculation within a multi-agent framework.

    This function executes a quantum chemistry calculation using QCEngine
    with parameters from the multi-agent state. It supports various quantum
    chemistry programs and calculation types.

    Parameters
    ----------
    state : MultiAgentState
        The current state of the multi-agent system containing calculation
        parameters

    Returns
    -------
    dict
        Dictionary containing the calculation results in a format suitable for
        the multi-agent system

    Raises
    ------
    ValueError
        If the calculator type is not supported
    Exception
        If the calculation fails or encounters an error

    Notes
    -----
    This function supports:
    - Single-point calculations (energy, gradient, hessian)
    - Geometry optimization
    - Vibrational frequency calculations
    - Thermochemistry calculations
    """
    parameters = state["parameter_response"][-1]
    params = json.loads(parameters.content)

    # Extract and contruct input for QCEngine
    qc_params = {}
    model = {}
    keywords = {}

    calculator = params["calculator"]

    calc_type = calculator["calculator_type"].lower()

    if calc_type == "psi4":
        from chemgraph.models.calculators.psi4_calc import Psi4Calc

        calc = Psi4Calc(**params["calculator"])
    elif calc_type == "mopac":
        from chemgraph.models.calculators.mopac_calc import MopacCalc

        calc = MopacCalc(**params["calculator"])
    else:
        raise ValueError(
            f"Unsupported calculator: {calculator}. Available calculators are Psi4 and MOPAC."
        )

    # Sort values to follow QCEngine's AtomicInput format
    calc_params = calc.model_dump()
    for item in list(calc_params):
        if item in ["method", "basis"]:
            model[item] = calc_params[item]
        elif item == "calculator_type":
            continue
        else:
            keywords[item] = calc_params[item]

    atomsdata = AtomsData(**params["atomsdata"])
    qc_params["molecule"] = convert_atomsdata_to_qcmolecule(atomsdata)
    qc_params["model"] = model
    qc_params["keywords"] = keywords
    qc_params["driver"] = params["driver"]

    if params["driver"] in [
        "energy",
        "gradient",
        "hessian",
    ]:  # Single-point calculations
        inp = qcel.models.AtomicInput(**qc_params)
        try:
            logger.info("Starting QCEngine calculation")
            result = qcengine.compute(inp, params["program"])
            # Handling vibrational frequency calculations
            if qc_params["driver"] == "hessian":
                from ase.data import atomic_masses

                masses = [atomic_masses[num] for num in params["atomsdata"]["numbers"]]
                print(result.dict())
                coords = result.molecule.geometry
                hessian = result.return_result
                print(
                    "INPUT VALUES FOR VIBRATIONAL FREQUENCIES: ",
                    masses,
                    coords,
                    hessian,
                )
                frequencies = compute_vibrational_frequencies(hessian, masses, coords)
                print("FREQUENCIES: ", frequencies)

            result = result.dict()
            if "stdout" in result:
                result.pop("stdout")
            output = []

            # Numpy encoder. Ensure the final results can be stored in LLM messages.
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()  # Convert NumPy array to list
                    if isinstance(obj, np.generic):
                        return obj.item()  # Convert NumPy scalar to Python scalar
                    return super().default(obj)

            output.append(HumanMessage(role="system", content=json.dumps(result, cls=NumpyEncoder)))
            logger.info("QCEngine calculation completed successfully")
            return {"opt_response": output}
        except Exception as e:
            logger.error(f"Error in QCEngine calculation: {str(e)}")
            return {"opt_response": output}

    elif params["driver"] in ["opt", "vib"]:
        from qcelemental.models import OptimizationInput
        from qcelemental.models.procedures import QCInputSpecification

        input_spec = QCInputSpecification(driver="gradient", model=model)
        opt_input = OptimizationInput(
            initial_molecule=convert_atomsdata_to_qcmolecule(atomsdata),
            input_specification=input_spec,
            protocols={"trajectory": "all"},
            keywords={"program": params["program"], "maxsteps": 100},
        )

        opt_input = {
            "initial_molecule": convert_atomsdata_to_qcmolecule(atomsdata),
            "input_specification": input_spec,
            "protocols": {"trajectory": "all"},
            "keywords": {"program": params["program"], "maxsteps": 100},
        }
        opt_output = qcengine.compute_procedure(opt_input, "geometric", raise_error=True)

        if params["driver"] == "vib":
            hess_inp = qcel.models.AtomicInput(
                molecule=opt_output.final_molecule,
                driver="hessian",
                model=model,
            )

            hess_output = qcengine.compute(hess_inp, params["program"])
            frequencies_in_cm = compute_vibrational_frequencies(
                hess_output.return_result,
                hess_output.molecule.masses,
                hess_output.molecule.geometry,
            )
            output_params = {
                "converged": hess_output.success,
                "final_structure": convert_qcmolecule_to_atomsdata(opt_output.final_molecule),
                "simulation_input": opt_input,
                "frequencies": list(frequencies_in_cm),
            }
            from chemgraph.models.qcengine_input import QCEngineOutput

            output = QCEngineOutput(**output_params)

            return output
