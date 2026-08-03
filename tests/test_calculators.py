import importlib.util
import sys

import pytest
import numpy as np
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.espresso_calc import EspressoCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.schemas.calculators.orca_calc import OrcaCalc
from chemgraph.schemas.calculators.tblite_calc import TBLiteCalc
from chemgraph.schemas.calculators.vasp_calc import VaspCalc
from ase import Atoms


@pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="TBLite not installed"
)
def test_xtb_alias_maps_to_tblite_calculator_schema():
    from chemgraph.schemas.ase_input import ASEInputSchema

    params = ASEInputSchema(
        input_structure_file="methane.xyz",
        driver="ir",
        calculator={"calculator_type": "xTB", "method": "GFN2-xTB"},
    )

    assert params.calculator.calculator_type == "TBLite"
    assert params.calculator.method == "GFN2-xTB"


def test_default_calculator_is_in_detected_available_calculators():
    from chemgraph.schemas.ase_input import (
        get_available_calculator_names,
        get_calculator_selection_context,
        get_default_calculator_name,
    )

    available = get_available_calculator_names()
    default = get_default_calculator_name()
    context = get_calculator_selection_context()

    assert default in available
    assert available
    assert "Calculator availability detected during ChemGraph initialization" in context
    assert default in context


def test_emt_calculator():
    # Test EMT calculator initialization
    calc = EMTCalc()
    ase_calc = calc.get_calculator()

    # Create a simple molecule
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.calc = ase_calc

    # Test energy calculation
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)

    # Test forces calculation
    forces = atoms.get_forces()
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 3)


@pytest.mark.skipif(
    importlib.util.find_spec("mace") is None, reason="MACE not installed"
)
def test_mace_calculator():
    # Test MACE calculator initialization
    calc = MaceCalc(model_type="medium")
    ase_calc = calc.get_calculator()

    # Create a simple molecule
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.calc = ase_calc

    # Test energy calculation
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)

    # Test forces calculation
    forces = atoms.get_forces()
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 3)


@pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="TBLite not installed"
)
def test_tblite_calculator():
    # Test TBLite calculator initialization
    calc = TBLiteCalc()
    ase_calc = calc.get_calculator()

    # Create a simple molecule
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.calc = ase_calc

    # Test energy calculation
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)

    # Test forces calculation
    forces = atoms.get_forces()
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 3)


@pytest.mark.skipif(
    importlib.util.find_spec("ase.io.orca") is None, reason="ORCA not installed"
)
def test_orca_calculator():
    # Test ORCA calculator initialization
    from ase.calculators.calculator import BadConfiguration
    from ase import Atoms

    try:
        calc = OrcaCalc()
        ase_calc = calc.get_calculator()
    except BadConfiguration:
        pytest.skip("ORCA calculator not configured in ASE.")

    # Create a simple molecule
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.calc = ase_calc

    # Test basic calculator properties
    assert hasattr(ase_calc, "calculate")


# ---------------------------------------------------------------------------
# VASP + Quantum ESPRESSO (subprocess DFT)
#
# These need a licensed/installed binary + pseudopotentials to *run*, which this
# machine lacks. But the ASE calculator objects *construct* without a binary, so
# we mock-test the schema -> ASE-object kwarg plumbing unconditionally, and gate
# only the availability-registration test on a monkeypatched environment.
# ---------------------------------------------------------------------------

_DFT_ENV_VARS = (
    "ASE_VASP_COMMAND",
    "VASP_COMMAND",
    "VASP_PP_PATH",
    "ASE_ESPRESSO_COMMAND",
    "ESPRESSO_PSEUDO",
)


def _available_calculators_in_subprocess(env_overrides):
    """Return get_available_calculator_names() from a fresh interpreter.

    The availability gating in ``chemgraph.schemas.ase_input`` runs once at
    import time, so testing it requires a pristine import under a controlled
    environment. We do that in a subprocess. Using ``importlib.reload`` would
    rebind ASEInputSchema/ASEOutputSchema to new class objects while
    ``chemgraph.tools.ase_core`` keeps its original references, which breaks
    pydantic identity checks for the rest of the session (cross-test leak).
    """
    import json
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k not in _DFT_ENV_VARS}
    env.update(env_overrides)
    script = (
        "import json;"
        "from chemgraph.schemas.ase_input import get_available_calculator_names;"
        "print(json.dumps(get_available_calculator_names()))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_vasp_calculator_builds_ase_object_with_kwargs():
    """VaspCalc maps its fields onto the ASE Vasp object's parameter stores."""
    calc = VaspCalc(
        encut=400.0,
        kpts=[2, 2, 2],
        ispin=2,
        nsw=5,
        ediffg=-0.02,
        prec="Accurate",
    )
    assert calc.get_multiplicity() is None  # VASP uses ispin, not 2S+1

    ase_calc = calc.get_calculator()
    assert type(ase_calc).__name__ == "Vasp"
    assert ase_calc.float_params["encut"] == 400.0
    assert ase_calc.int_params["ispin"] == 2
    assert ase_calc.int_params["nsw"] == 5
    assert ase_calc.exp_params["ediffg"] == -0.02
    assert ase_calc.string_params["prec"] == "Accurate"
    assert ase_calc.input_params["kpts"] == [2, 2, 2]


def test_vasp_charge_and_setups_reach_ase():
    """charge and setups land in the ASE Vasp input_params store."""
    ase_calc = VaspCalc(charge=-1, setups="recommended").get_calculator()
    assert ase_calc.input_params["charge"] == -1
    assert ase_calc.input_params["setups"] == "recommended"


def test_vasp_kspacing_replaces_kpts():
    """When kspacing is given it wins and kpts is not passed through."""
    ase_calc = VaspCalc(kspacing=0.25, kpts=[4, 4, 4]).get_calculator()
    assert ase_calc.float_params["kspacing"] == 0.25
    # kpts falls back to the ASE default (not the schema's [4,4,4]).
    assert ase_calc.input_params["kpts"] != [4, 4, 4]


def test_vasp_rejects_wrong_calculator_type():
    with pytest.raises(ValueError):
        VaspCalc(calculator_type="notvasp").get_calculator()


def test_espresso_calculator_builds_ase_object_with_kwargs():
    """EspressoCalc assembles the pw.x input_data namelist + passes k-points."""
    calc = EspressoCalc(
        ecutwfc=40.0,
        ecutrho=320.0,
        kpts=[2, 2, 2],
        xc="PBE",
        smearing="mv",
        degauss=0.01,
        pseudopotentials={"Si": "Si.UPF"},
        pseudo_dir="/tmp/pseudo",
    )
    assert calc.get_multiplicity() is None  # QE uses nspin, not 2S+1

    ase_calc = calc.get_calculator()
    assert type(ase_calc).__name__ == "Espresso"
    input_data = ase_calc.parameters["input_data"]
    assert input_data["ecutwfc"] == 40.0
    assert input_data["ecutrho"] == 320.0
    assert input_data["input_dft"] == "PBE"
    assert input_data["occupations"] == "smearing"
    assert input_data["smearing"] == "mv"
    assert input_data["degauss"] == 0.01
    assert ase_calc.parameters["kpts"] == (2, 2, 2)
    assert ase_calc.parameters["pseudopotentials"] == {"Si": "Si.UPF"}


def test_espresso_input_data_override_wins():
    """Raw input_data keys override the convenience-field defaults."""
    calc = EspressoCalc(ecutwfc=40.0, input_data={"ecutwfc": 80.0, "nbnd": 20})
    data = calc._build_input_data()
    assert data["ecutwfc"] == 80.0  # override wins over the field
    assert data["nbnd"] == 20


def test_espresso_requests_forces_by_default():
    """pw.x must print forces every SCF: ChemGraph drives geometry with an ASE
    optimizer, so QE is a force engine. ASE 3.29 does not translate
    properties=['forces'] into the 'tprnfor' control flag, so EspressoCalc must
    set it or the first optimizer step raises 'forces not present'."""
    data = EspressoCalc(ecutwfc=25.0)._build_input_data()
    assert data["tprnfor"] is True


def test_espresso_tprnfor_is_overridable():
    """A user can still turn force printing off via raw input_data."""
    calc = EspressoCalc(ecutwfc=25.0, input_data={"tprnfor": False})
    assert calc._build_input_data()["tprnfor"] is False


def test_espresso_rejects_wrong_calculator_type():
    with pytest.raises(ValueError):
        EspressoCalc(calculator_type="notqe").get_calculator()


@pytest.mark.parametrize(
    "env, expected",
    [
        (
            {
                "ASE_VASP_COMMAND": "vasp_std",
                "VASP_PP_PATH": "/tmp/pp",
                "ASE_ESPRESSO_COMMAND": "pw.x",
                "ESPRESSO_PSEUDO": "/tmp/pseudo",
            },
            True,
        ),
        ({}, False),  # nothing configured -> both gated off
    ],
)
def test_subprocess_dft_availability_gating(env, expected):
    """VaspCalc/EspressoCalc register only when their binary + pseudos are set."""
    names = _available_calculators_in_subprocess(env)
    assert ("VaspCalc" in names) is expected
    assert ("EspressoCalc" in names) is expected


@pytest.mark.parametrize("alias", ["espresso", "qe", "pwscf", "pw"])
def test_espresso_aliases_route_to_espresso_calc(alias):
    """load_calculator routes espresso/qe/pwscf/pw to EspressoCalc + ASE Espresso.

    This exercises the dispatch + alias normalization directly (no binary, no
    module reload), so it stays hermetic and isolated.
    """
    from chemgraph.tools.ase_core import load_calculator

    ase_calc, _extra, model = load_calculator(
        {"calculator_type": alias, "ecutwfc": 30.0, "pseudo_dir": "/tmp/pseudo"}
    )
    assert type(model).__name__ == "EspressoCalc"
    assert type(ase_calc).__name__ == "Espresso"
    # Alias is normalized to the canonical type so get_calculator's check passes.
    assert model.calculator_type == "espresso"
    assert model.ecutwfc == 30.0


@pytest.mark.parametrize("alias", ["vasp", "VASP", "Vasp", "vasp_std"])
def test_vasp_aliases_route_to_vasp_calc(alias):
    """load_calculator routes vasp/VASP/Vasp/vasp_std to VaspCalc + ASE Vasp.

    Guards against the normalization asymmetry where a non-canonical
    calculator_type reached VaspCalc.get_calculator's strict check and raised.
    """
    from chemgraph.tools.ase_core import load_calculator

    ase_calc, _extra, model = load_calculator(
        {"calculator_type": alias, "encut": 300.0, "kpts": [1, 1, 1]}
    )
    assert type(model).__name__ == "VaspCalc"
    assert type(ase_calc).__name__ == "Vasp"
    assert model.calculator_type == "vasp"  # normalized to canonical
    assert model.encut == 300.0


# ---------------------------------------------------------------------------
# PR1 DFT-correctness fixes: pbc-aware k-mesh, QE command sanitization,
# VASP INCAR escape hatch, and molecule centering. All hermetic (kwargs -> ASE
# object / written file); no pw.x or VASP binary is invoked.
# ---------------------------------------------------------------------------


def _h2o_molecule():
    """A non-periodic water molecule (pbc all False, no cell)."""
    from ase.build import molecule

    return molecule("H2O")


def _si_bulk():
    """A periodic silicon crystal (pbc all True, finite cell)."""
    from ase.build import bulk

    return bulk("Si")


def _cu_slab():
    """A metal slab, periodic in x/y and vacuum in z (pbc=[T,T,F])."""
    from ase.build import fcc111

    return fcc111("Cu", size=(1, 1, 3), vacuum=8.0)


def _cu_wire():
    """A 1D wire, periodic in x only (pbc=[T,F,F])."""
    from ase.build import bulk

    wire = bulk("Cu")
    wire.pbc = [True, False, False]
    return wire


# --- Fix 1: pbc-aware k-mesh --------------------------------------------------


def test_espresso_molecule_drops_kmesh_for_gamma():
    """A non-periodic molecule must get NO k-mesh so ASE writes K_POINTS gamma;
    a Monkhorst-Pack mesh is meaningless for an isolated molecule."""
    ase_calc = EspressoCalc(
        pseudopotentials={"O": "O.UPF", "H": "H.UPF"}
    ).get_calculator(atoms=_h2o_molecule())
    assert "kpts" not in ase_calc.parameters
    assert "kspacing" not in ase_calc.parameters


def test_espresso_periodic_keeps_configured_mesh():
    """A periodic solid keeps its configured Monkhorst-Pack mesh."""
    ase_calc = EspressoCalc().get_calculator(atoms=_si_bulk())
    assert ase_calc.parameters["kpts"] == (4, 4, 4)


def test_espresso_atoms_none_keeps_mesh_backward_compat():
    """atoms=None (construction / existing no-arg callers) keeps the mesh."""
    ase_calc = EspressoCalc().get_calculator()
    assert ase_calc.parameters["kpts"] == (4, 4, 4)


def test_vasp_molecule_pins_single_gamma_point():
    """A non-periodic molecule pins kpts=[1,1,1]+gamma (real VASP needs a
    KPOINTS file, so None is wrong) and drops kspacing (an INCAR KSPACING would
    silently override the single Gamma point)."""
    ase_calc = VaspCalc(kspacing=0.25).get_calculator(atoms=_h2o_molecule())
    assert ase_calc.input_params["kpts"] == [1, 1, 1]
    assert ase_calc.input_params["gamma"] is True
    assert ase_calc.float_params["kspacing"] is None


def test_vasp_periodic_keeps_configured_mesh():
    """A periodic solid keeps its configured Monkhorst-Pack mesh."""
    ase_calc = VaspCalc().get_calculator(atoms=_si_bulk())
    assert ase_calc.input_params["kpts"] == [3, 3, 3]


def test_vasp_atoms_none_keeps_mesh_backward_compat():
    """atoms=None keeps the configured mesh (backward compatible)."""
    ase_calc = VaspCalc().get_calculator()
    assert ase_calc.input_params["kpts"] == [3, 3, 3]


def test_vasp_gamma_field_threads_for_periodic():
    """The gamma field requests a Gamma-centered mesh for a periodic solid."""
    ase_calc = VaspCalc(gamma=True).get_calculator(atoms=_si_bulk())
    assert ase_calc.input_params["gamma"] is True


def test_vasp_molecule_defaults_gaussian_smearing():
    """A non-periodic molecule defaults to ismear=0 (+small sigma) so it does not
    inherit VASP's metallic default ISMEAR=1 (spurious partial occupancies)."""
    ase_calc = VaspCalc().get_calculator(atoms=_h2o_molecule())
    assert ase_calc.int_params["ismear"] == 0
    assert ase_calc.float_params["sigma"] == 0.03


def test_vasp_molecule_smearing_overridable_via_input_data():
    """The molecule ismear default is a convenience, still overridable by the
    input_data escape hatch (merged last)."""
    ase_calc = VaspCalc(
        input_data={"ismear": -1, "sigma": 0.2}
    ).get_calculator(atoms=_h2o_molecule())
    assert ase_calc.int_params["ismear"] == -1
    assert ase_calc.float_params["sigma"] == 0.2


# --- Fix D: per-axis k-mesh for slabs / wires ---------------------------------


def test_espresso_slab_masks_vacuum_axis():
    """A slab (pbc=[T,T,F]) keeps its in-plane mesh but collapses the vacuum
    axis to a single k-point."""
    ase_calc = EspressoCalc(kpts=[6, 6, 6]).get_calculator(atoms=_cu_slab())
    assert ase_calc.parameters["kpts"] == (6, 6, 1)


def test_espresso_wire_masks_two_vacuum_axes():
    """A wire (pbc=[T,F,F]) keeps only its periodic axis subdivided."""
    ase_calc = EspressoCalc(kpts=[8, 8, 8]).get_calculator(atoms=_cu_wire())
    assert ase_calc.parameters["kpts"] == (8, 1, 1)


def test_vasp_slab_masks_vacuum_axis():
    """VASP slab (pbc=[T,T,F]) collapses the vacuum axis to one k-point."""
    ase_calc = VaspCalc(kpts=[6, 6, 6]).get_calculator(atoms=_cu_slab())
    assert ase_calc.input_params["kpts"] == [6, 6, 1]


def test_vasp_wire_masks_two_vacuum_axes():
    """VASP wire (pbc=[T,F,F]) keeps only the periodic axis subdivided."""
    ase_calc = VaspCalc(kpts=[8, 8, 8]).get_calculator(atoms=_cu_wire())
    assert ase_calc.input_params["kpts"] == [8, 1, 1]


def test_mask_kmesh_by_pbc_helper():
    """The shared per-axis mask helper is exercised directly."""
    from chemgraph.schemas.calculators._plane_wave import mask_kmesh_by_pbc

    assert mask_kmesh_by_pbc([4, 4, 4], [True, True, True]) == (4, 4, 4)
    assert mask_kmesh_by_pbc([4, 4, 4], [True, True, False]) == (4, 4, 1)
    assert mask_kmesh_by_pbc([4, 4, 4], [True, False, False]) == (4, 1, 1)
    assert mask_kmesh_by_pbc([4, 4, 4], [False, False, False]) == (1, 1, 1)


# --- Fix 2: QE command sanitization ------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("pw.x", "pw.x"),
        ("mpirun -np 4 pw.x -in PREFIX.pwi > PREFIX.pwo", "mpirun -np 4 pw.x"),
        ("srun pw.x -inp foo.pwi", "srun pw.x"),
        ("mpirun -np 8 pw.x < in.pwi", "mpirun -np 8 pw.x"),
        ("pw.x -in a.pwi >> log", "pw.x"),
        # runtime flags (npool/ndiag) are launch args, not ASE's -in -> kept
        ("pw.x -npool 4 -ndiag 1", "pw.x -npool 4 -ndiag 1"),
    ],
)
def test_espresso_command_sanitizer(raw, expected):
    """Legacy full-command ASE_ESPRESSO_COMMAND is reduced to its launch prefix;
    ASE appends '-in <file>' itself so the input/redirection tail must be cut."""
    from chemgraph.schemas.calculators.espresso_calc import (
        _sanitize_espresso_command,
    )

    assert _sanitize_espresso_command(raw) == expected


def test_espresso_profile_command_is_sanitized(monkeypatch):
    """The EspressoProfile built from a legacy env value carries only the clean
    launch prefix, not the duplicated -in / redirection tokens."""
    monkeypatch.setenv(
        "ASE_ESPRESSO_COMMAND", "mpirun -np 4 pw.x -in PREFIX.pwi > PREFIX.pwo"
    )
    ase_calc = EspressoCalc(pseudopotentials={"O": "O.UPF"}).get_calculator()
    assert ase_calc.profile.command == "mpirun -np 4 pw.x"


# --- Fix 3: VASP INCAR escape hatch ------------------------------------------


def test_vasp_input_data_reach_ase_param_stores():
    """Advanced INCAR tags land in their correct ASE parameter groups. Note
    lreal routes to special_params (NOT string_params)."""
    ase_calc = VaspCalc(
        input_data={
            "ismear": 0,
            "sigma": 0.05,
            "algo": "Fast",
            "lreal": "Auto",
        }
    ).get_calculator(atoms=_si_bulk())
    assert ase_calc.int_params["ismear"] == 0
    assert ase_calc.float_params["sigma"] == 0.05
    assert ase_calc.string_params["algo"] == "Fast"
    assert ase_calc.special_params["lreal"] == "Auto"


def test_vasp_input_data_override_convenience_field():
    """input_data is merged last, so it wins over a convenience field."""
    ase_calc = VaspCalc(
        encut=400.0, input_data={"encut": 650.0}
    ).get_calculator(atoms=_si_bulk())
    assert ase_calc.float_params["encut"] == 650.0


def test_vasp_input_data_uppercase_keys_are_lowercased():
    """A user pasting canonical UPPERCASE INCAR names (as they appear in a real
    INCAR file) must still reach ASE -- keys are lowercased on merge, so
    {'ISMEAR': 0} is not silently dropped."""
    ase_calc = VaspCalc(
        input_data={"ISMEAR": 0, "SIGMA": 0.05}
    ).get_calculator(atoms=_si_bulk())
    assert ase_calc.int_params["ismear"] == 0
    assert ase_calc.float_params["sigma"] == 0.05


# --- Fix 0: molecule centering (via the ase_core runner seam) -----------------


def test_prepare_dft_calculator_centers_cell_less_molecule_and_writes_gamma(
    tmp_path,
):
    """The real ase_core seam (prepare_dft_calculator) centers a cell-less
    non-periodic molecule (so the plane-wave writer has a finite box) while
    leaving pbc False (so it stays on the Gamma path), and the written QE input
    carries K_POINTS gamma + a real non-zero cell. Exercises the shipped helper,
    not a re-implementation -- reverting the seam turns this red."""
    from ase.io.espresso import write_espresso_in

    from chemgraph.tools.ase_core import load_calculator, prepare_dft_calculator

    atoms = _h2o_molecule()
    assert atoms.cell.rank == 0 and not atoms.pbc.any()

    calc, _extra, model = load_calculator(
        {
            "calculator_type": "espresso",
            "pseudopotentials": {"O": "O.UPF", "H": "H.UPF"},
        }
    )
    ase_calc = prepare_dft_calculator(atoms, calc, model)

    # The helper centered the molecule in place but must NOT turn on periodicity.
    assert atoms.cell.rank == 3
    assert not atoms.pbc.any()

    out = tmp_path / "espresso.pwi"
    with open(out, "w") as fh:
        write_espresso_in(
            fh,
            atoms,
            input_data=ase_calc.parameters["input_data"],
            pseudopotentials={"O": "O.UPF", "H": "H.UPF"},
        )
    text = out.read_text()
    assert "K_POINTS gamma" in text
    # A real box: the centered cell diagonal must be non-zero, proving the
    # centering reached the writer (a param-only test can miss a zero cell).
    assert "CELL_PARAMETERS" in text
    cell_lines = text.split("CELL_PARAMETERS", 1)[1].splitlines()[1:4]
    diag = [float(cell_lines[i].split()[i]) for i in range(3)]
    assert all(d > 1.0 for d in diag)


def test_prepare_dft_calculator_rebuilds_qe_gamma_only(tmp_path):
    """prepare_dft_calculator rebuilds the QE calculator Gamma-only for a
    cell-less molecule read from disk (the run_ase_core code path)."""
    from ase.io import read, write

    from chemgraph.tools.ase_core import load_calculator, prepare_dft_calculator

    xyz = tmp_path / "h2o.xyz"
    write(str(xyz), _h2o_molecule())

    calc, _extra, model = load_calculator(
        {"calculator_type": "espresso", "pseudo_dir": "/tmp/pseudo"}
    )
    assert type(model).__name__ == "EspressoCalc"

    atoms = read(str(xyz))
    rebuilt = prepare_dft_calculator(atoms, calc, model)
    assert "kpts" not in rebuilt.parameters  # Gamma-only for the molecule


def test_prepare_dft_calculator_is_noop_for_non_dft(tmp_path):
    """The isinstance guard leaves a non-DFT calculator (EMT) untouched: the
    same object is returned and a cell-less molecule is NOT centered."""
    from chemgraph.tools.ase_core import load_calculator, prepare_dft_calculator

    calc, _extra, model = load_calculator({"calculator_type": "emt"})
    atoms = _h2o_molecule()
    result = prepare_dft_calculator(atoms, calc, model)
    assert result is calc
    assert atoms.cell.rank == 0  # not centered -- EMT needs no cell


def test_prepare_dft_calculator_keeps_user_supplied_full_cell():
    """A non-periodic molecule that ALREADY carries a full rank-3 cell must not
    be re-centered -- a user-supplied box is preserved. Guards the ``rank < 3``
    condition: dropping it would clobber the box with a fresh vacuum cell."""
    import numpy as np

    from chemgraph.tools.ase_core import load_calculator, prepare_dft_calculator

    atoms = _h2o_molecule()
    atoms.center(vacuum=9.0)  # rank-3 cell, pbc still all False
    assert atoms.cell.rank == 3 and not atoms.pbc.any()
    preset = atoms.cell.array.copy()

    calc, _extra, model = load_calculator({"calculator_type": "espresso"})
    prepare_dft_calculator(atoms, calc, model)

    assert np.allclose(atoms.cell.array, preset)  # untouched


def test_prepare_dft_calculator_centers_degenerate_partial_cell():
    """A non-periodic molecule with a degenerate (rank 1/2, zero-volume) cell is
    still given a clean vacuum box -- a singular lattice vector would make pw.x/
    VASP emit garbage. Guards the ``rank < 3`` condition on the low side."""
    from chemgraph.tools.ase_core import load_calculator, prepare_dft_calculator

    atoms = _h2o_molecule()
    atoms.cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 0]]  # rank-2, one zero vector
    atoms.pbc = [False, False, False]
    assert atoms.cell.rank == 2

    calc, _extra, model = load_calculator({"calculator_type": "espresso"})
    prepare_dft_calculator(atoms, calc, model)

    assert atoms.cell.rank == 3  # centered into a finite, non-singular box
    assert not atoms.pbc.any()


# --- Fix follow-ups: kpts length validation ----------------------------------


@pytest.mark.parametrize("Calc", [EspressoCalc, VaspCalc])
@pytest.mark.parametrize("bad_kpts", [[4, 4], [4, 4, 4, 4], []])
def test_plane_wave_kpts_must_have_three_axes(Calc, bad_kpts):
    """kpts must be exactly 3 axes; a 2- or 4-element mesh is rejected at
    construction, so it cannot crash later in the ASE writer (IndexError)."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Calc(kpts=bad_kpts)


def test_is_nonperiodic_helper():
    """is_nonperiodic: True only for a real, fully non-periodic structure."""
    from chemgraph.schemas.calculators._plane_wave import is_nonperiodic

    assert is_nonperiodic(_h2o_molecule()) is True
    assert is_nonperiodic(_si_bulk()) is False
    assert is_nonperiodic(_cu_slab()) is False
    assert is_nonperiodic(None) is False  # construction time -> use configured mesh


# --- Fix follow-ups: QE kspacing dropped for a molecule ----------------------


def test_espresso_molecule_drops_kspacing():
    """A fully non-periodic molecule must not carry a kspacing (which would make
    ASE build a Monkhorst-Pack mesh); it stays on the K_POINTS gamma path.
    Parallels the VASP kspacing-drop coverage."""
    ase_calc = EspressoCalc(kspacing=0.25).get_calculator(atoms=_h2o_molecule())
    assert "kspacing" not in ase_calc.parameters
    assert "kpts" not in ase_calc.parameters


# ---------------------------------------------------------------------------
# run_qe / run_vasp: calculator-pinned wrappers over run_ase.
#
# These are hermetic: they exercise the pinned schemas + the dispatch parity
# with run_ase, never invoking a pw.x / VASP binary. The pinned schemas
# deliberately skip the availability gate so they construct on this machine.
# ---------------------------------------------------------------------------


def test_qe_input_schema_pins_espresso_calculator():
    """QEInputSchema coerces a bare dict into EspressoCalc (canonical type),
    constructible even though pw.x is not installed here."""
    from chemgraph.schemas.plane_wave_input import QEInputSchema

    params = QEInputSchema(
        input_structure_file="si.cif",
        calculator={"ecutwfc": 40.0, "kpts": [2, 2, 2]},
    )
    assert type(params.calculator).__name__ == "EspressoCalc"
    assert params.calculator.calculator_type == "espresso"
    assert params.calculator.ecutwfc == 40.0


def test_qe_input_schema_default_calculator():
    """With no calculator payload QEInputSchema defaults to a plain EspressoCalc."""
    from chemgraph.schemas.plane_wave_input import QEInputSchema

    params = QEInputSchema(input_structure_file="si.cif")
    assert type(params.calculator).__name__ == "EspressoCalc"
    assert params.calculator.calculator_type == "espresso"


def test_qe_input_schema_accepts_qe_alias():
    """A QE alias in calculator_type is dropped in favor of the canonical tag,
    so ase_core dispatch always sees 'espresso'."""
    from chemgraph.schemas.plane_wave_input import QEInputSchema

    params = QEInputSchema(
        input_structure_file="si.cif",
        calculator={"calculator_type": "qe", "ecutwfc": 30.0},
    )
    assert params.calculator.calculator_type == "espresso"


def test_vasp_input_schema_pins_vasp_calculator():
    """VaspInputSchema coerces a bare dict into VaspCalc (canonical type)."""
    from chemgraph.schemas.plane_wave_input import VaspInputSchema

    params = VaspInputSchema(
        input_structure_file="si.cif",
        calculator={"encut": 400.0, "kpts": [2, 2, 2]},
    )
    assert type(params.calculator).__name__ == "VaspCalc"
    assert params.calculator.calculator_type == "vasp"
    assert params.calculator.encut == 400.0


def test_vasp_input_schema_rejects_wrong_instance():
    """A pre-built calculator of the wrong type is a caller error."""
    from chemgraph.schemas.plane_wave_input import VaspInputSchema

    with pytest.raises(ValueError):
        VaspInputSchema(
            input_structure_file="si.cif", calculator=EspressoCalc()
        )


def test_qe_input_schema_rejects_wrong_instance():
    """The mirror of the VASP case: a VaspCalc handed to QEInputSchema raises."""
    from chemgraph.schemas.plane_wave_input import QEInputSchema

    with pytest.raises(ValueError):
        QEInputSchema(input_structure_file="si.cif", calculator=VaspCalc())


def test_run_qe_tool_schema_excludes_vasp_fields():
    """run_qe's tool schema exposes only QE parameters, not the full calculator
    union -- the whole point of the pinned tool (smaller per-call prompt)."""
    import json

    from langchain_core.utils.function_calling import convert_to_openai_tool

    from chemgraph.tools.ase_tools import run_qe

    blob = json.dumps(convert_to_openai_tool(run_qe))
    assert "ecutwfc" in blob  # QE field present
    assert "encut" not in blob  # VASP-only field absent


def test_run_vasp_tool_schema_excludes_qe_fields():
    """The mirror of the QE case: run_vasp's schema exposes only VASP
    parameters, not QE-only fields."""
    import json

    from langchain_core.utils.function_calling import convert_to_openai_tool

    from chemgraph.tools.ase_tools import run_vasp

    blob = json.dumps(convert_to_openai_tool(run_vasp))
    assert "encut" in blob  # VASP field present
    assert "ecutwfc" not in blob  # QE-only field absent


def test_run_qe_delegates_espresso_payload_to_core(tmp_path, monkeypatch):
    """run_qe hands run_ase_core an EspressoCalc payload with calculator_type
    'espresso', the same shape run_ase produces for an espresso calculator on a
    host where QE is installed. We stub run_ase_core (no pw.x) and inspect the
    params it receives. Dispatch routing of that payload is covered separately by
    test_espresso_aliases_route_to_espresso_calc."""
    import chemgraph.tools.ase_tools as ase_tools

    captured = {}

    def _capture(params):
        captured["type"] = type(params).__name__
        captured["calc"] = params.calculator.model_dump()
        captured["driver"] = params.driver
        return {"success": True}

    monkeypatch.setattr(ase_tools, "run_ase_core", _capture)

    ase_tools.run_qe.func(
        ase_tools.QEInputSchema(
            input_structure_file=str(tmp_path / "si.cif"),
            driver="energy",
            calculator={"ecutwfc": 35.0},
        )
    )

    assert captured["type"] == "QEInputSchema"
    assert captured["calc"]["calculator_type"] == "espresso"
    assert captured["calc"]["ecutwfc"] == 35.0
    assert captured["driver"] == "energy"


def test_run_vasp_delegates_vasp_payload_to_core(tmp_path, monkeypatch):
    """run_vasp hands run_ase_core a VaspCalc payload with calculator_type
    'vasp' (stubbed core, no VASP binary). Dispatch routing is covered by
    test_vasp_aliases_route_to_vasp_calc."""
    import chemgraph.tools.ase_tools as ase_tools

    captured = {}

    def _capture(params):
        captured["type"] = type(params).__name__
        captured["calc"] = params.calculator.model_dump()
        captured["driver"] = params.driver
        return {"success": True}

    monkeypatch.setattr(ase_tools, "run_ase_core", _capture)

    ase_tools.run_vasp.func(
        ase_tools.VaspInputSchema(
            input_structure_file=str(tmp_path / "si.cif"),
            driver="energy",
            calculator={"encut": 500.0},
        )
    )

    assert captured["type"] == "VaspInputSchema"
    assert captured["calc"]["calculator_type"] == "vasp"
    assert captured["calc"]["encut"] == 500.0
    assert captured["driver"] == "energy"


def test_plane_wave_tools_registered_only_when_engine_available(monkeypatch):
    """_plane_wave_tools offers run_qe/run_vasp only for detected engines, so an
    uninstalled engine is never bound into the agent (nor its token cost paid)."""
    import chemgraph.graphs.single_agent as sa

    monkeypatch.setattr(
        sa, "get_available_calculator_names", lambda: ["EMTCalc", "EspressoCalc"]
    )
    names = {t.name for t in sa._plane_wave_tools()}
    assert names == {"run_qe"}

    monkeypatch.setattr(
        sa,
        "get_available_calculator_names",
        lambda: ["EMTCalc", "EspressoCalc", "VaspCalc"],
    )
    names = {t.name for t in sa._plane_wave_tools()}
    assert names == {"run_qe", "run_vasp"}

    monkeypatch.setattr(sa, "get_available_calculator_names", lambda: ["EMTCalc"])
    assert sa._plane_wave_tools() == []


# --- Fix follow-ups: sanitizer edge branches ---------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # positional input-file operand with no preceding -in flag
        ("pw.x in.pwi", "pw.x"),
        ("mpirun -np 4 pw.x job.pwo", "mpirun -np 4 pw.x"),
        # unbalanced quote -> shlex.split raises -> raw (stripped) fallback
        ('pw.x "unterminated', 'pw.x "unterminated'),
        # command that is nothing but a stop token -> raw (stripped) fallback
        ("-in only.pwi", "-in only.pwi"),
        # whitespace-only -> empty
        ("   ", ""),
    ],
)
def test_espresso_command_sanitizer_edge_branches(raw, expected):
    """The fallback branches: a bare positional operand is cut, an unparseable
    string or an all-stripped result falls back to the raw (stripped) input."""
    from chemgraph.schemas.calculators.espresso_calc import (
        _sanitize_espresso_command,
    )

    assert _sanitize_espresso_command(raw) == expected
