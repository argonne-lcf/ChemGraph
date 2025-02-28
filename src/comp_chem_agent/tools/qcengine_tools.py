# from comp_chem_agent.state.state import MultiAgentState
from comp_chem_agent.state.opt_vib_state import MultiAgentState
from langchain_core.messages import HumanMessage
import json
from langchain.tools import tool
import qcengine
import numpy as np
from comp_chem_agent.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@tool
def run_qcengine(state: MultiAgentState, program="psi4"):
    """Run a QCEngine calculation.

    Args:
        state: The state of the multi-agent system.
        program: The program to use for the calculation.
    """
    atomic_numbers = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
    }

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
        output.append(
            HumanMessage(role="system", content=json.dumps(result, cls=NumpyEncoder))
        )
        logger.info("QCEngine calculation completed successfully")
        return {"opt_response": output}
    except Exception as e:
        logger.error(f"Error in QCEngine calculation: {str(e)}")
        raise


def is_linear_molecule(coords, tol=1e-3):
    """
    Determine if a molecule is linear.

    Parameters
    ----------
    coords : np.ndarray
        (N x 3) array of Cartesian coordinates.
    tol : float
        Tolerance for linearity; if the ratio of the second largest to largest
        singular value is below tol, the molecule is considered linear.

    Returns
    -------
    bool
        True if the molecule is linear, False otherwise.
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
    """
    Mass-weights the Hessian matrix.

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
    """
    Build a projection operator that projects out the translational and rotational
    degrees of freedom.

    Parameters
    ----------
    masses : array-like
        List of atomic masses (in amu) for N atoms.
    coords : np.ndarray
        (N x 3) array of Cartesian coordinates.
    is_linear : bool
        If True, the molecule is assumed linear (removes 3 translations and 2 rotations);
        otherwise, it removes 3 translations and 3 rotations.

    Returns
    -------
    P : np.ndarray
        A (3N x 3N) projection matrix.
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
    """
    Calculate vibrational frequencies (in cm⁻¹) from a Hessian matrix and atomic masses.
    Optionally projects out translational and rotational modes if coordinates are provided.

    For non-linear molecules, 3 translational and 3 rotational modes are removed.
    For linear molecules (including diatomics), 3 translational and 2 rotational modes are removed.

    Parameters
    ----------
    hessian : np.ndarray
        A (3N x 3N) Hessian matrix in atomic units.
    masses : array-like
        List/array of atomic masses in amu.
    coords : np.ndarray or None
        (N x 3) Cartesian coordinates. If provided, used to project out rotation and translation.
    linear : bool or None
        If set, explicitly specifies whether the molecule is linear.
        If None and coords is provided, linearity is determined automatically.

    Returns
    -------
    frequencies_cm : np.ndarray
        Array of vibrational frequencies in cm⁻¹ (negative values indicate imaginary modes).
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
