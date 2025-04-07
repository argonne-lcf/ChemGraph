import subprocess
import os
from pathlib import Path
import shutil
import numpy as np
import ase
from comp_chem_agent.models.graspa_input import GRASPAInputSchema
from langchain_core.tools import tool


_file_dir = Path(__file__).parent / "files" / "graspa_sycl_template"


@tool
def run_graspa(graspa_input: GRASPAInputSchema):
    """Run a gRASPA simulation and return the uptakes and errors.

    Args:
        graspa_input (str): a GRASPAInputSchema.

    Returns:
        Computed uptake (U) and error (E) from gRASPA-sycl in the following order:
            - U (mol/kg)
            - E (mol/kg)
            - U (g/L)
            - E (g/L)
    """

    def _calculate_cell_size(atoms: ase.Atoms, cutoff: float = 12.8) -> list[int, int, int]:
        """Method to calculate Unitcells (for periodic boundary condition) for GCMC

        Args:
            atoms (ase.Atoms): ASE atom object
            cutoff (float, optional): Cutoff in Angstrom. Defaults to 12.8.

        Returns:
            list[int, int, int]: Unit cell in x, y and z
        """
        unit_cell = atoms.cell[:]
        # Unit cell vectors
        a = unit_cell[0]
        b = unit_cell[1]
        c = unit_cell[2]
        # minimum distances between unit cell faces
        wa = np.divide(
            np.linalg.norm(np.dot(np.cross(b, c), a)),
            np.linalg.norm(np.cross(b, c)),
        )
        wb = np.divide(
            np.linalg.norm(np.dot(np.cross(c, a), b)),
            np.linalg.norm(np.cross(c, a)),
        )
        wc = np.divide(
            np.linalg.norm(np.dot(np.cross(a, b), c)),
            np.linalg.norm(np.cross(a, b)),
        )

        uc_x = int(np.ceil(cutoff / (0.5 * wa)))
        uc_y = int(np.ceil(cutoff / (0.5 * wb)))
        uc_z = int(np.ceil(cutoff / (0.5 * wc)))

        return [uc_x, uc_y, uc_z]

    def _write_cif(atoms: ase.Atoms, out_dir: str, name: str):
        """Save a CIF file with partial charges for gRASPA-sycl from an ASE Atoms object.

        Args:
            atoms (ase.Atoms): ASE Atoms object containing partial charges in atoms.info["_atom_site_charge"].
            out_dir (str): Directory to save the output file.
            name (str): Name of the output file.
        """

        with open(os.path.join(out_dir, name), "w") as fp:
            fp.write(f"MOFA-{name}\n")

            a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            fp.write(f"_cell_length_a      {a}\n")
            fp.write(f"_cell_length_b      {b}\n")
            fp.write(f"_cell_length_c      {c}\n")
            fp.write(f"_cell_angle_alpha   {alpha}\n")
            fp.write(f"_cell_angle_beta    {beta}\n")
            fp.write(f"_cell_angle_gamma   {gamma}\n")
            fp.write("\n")
            fp.write("_symmetry_space_group_name_H-M    'P 1'\n")
            fp.write("_symmetry_int_tables_number        1\n")
            fp.write("\n")

            fp.write("loop_\n")
            fp.write("  _symmetry_equiv_pos_as_xyz\n")
            fp.write("  'x, y, z'\n")
            fp.write("loop_\n")
            fp.write("  _atom_site_label\n")
            fp.write("  _atom_site_occupancy\n")
            fp.write("  _atom_site_fract_x\n")
            fp.write("  _atom_site_fract_y\n")
            fp.write("  _atom_site_fract_z\n")
            fp.write("  _atom_site_thermal_displace_type\n")
            fp.write("  _atom_site_B_iso_or_equiv\n")
            fp.write("  _atom_site_type_symbol\n")
            fp.write("  _atom_site_charge\n")

            coords = atoms.get_scaled_positions().tolist()
            symbols = atoms.get_chemical_symbols()
            occupancies = [1 for i in range(len(symbols))]  # No partial occupancy
            charges = atoms.info["_atom_site_charge"]

            no = {}

            for symbol, pos, occ, charge in zip(symbols, coords, occupancies, charges):
                if symbol in no:
                    no[symbol] += 1
                else:
                    no[symbol] = 1

                fp.write(
                    f"{symbol}{no[symbol]} {occ:.1f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} Biso 1.0 {symbol} {charge:.6f}\n"
                )

    def _get_cif_from_chargemol(
        cp2k_path: str,
        chargemol_fname: str = "DDEC6_even_tempered_net_atomic_charges.xyz",
    ) -> ase.Atoms:
        """Return an ASE atom object from a Chargemol output file.

        Args:
            cp2k_path (str): Path to Chargemol output.
            chargemol_fname (str, optional): Chargemol output filename. Defaults to "DDEC6_even_tempered_net_atomic_charges.xyz".

        Returns:
            ase.Atoms: ASE Atoms object containing partial charges in atoms.info["_atom_site_charge"].
        """
        with open(os.path.join(cp2k_path, chargemol_fname), "r") as f:
            symbols = []
            x = []
            y = []
            z = []
            charges = []
            positions = []
            lines = f.readlines()

            for i, line in enumerate(lines):
                if i == 0:
                    natoms = int(line)
                elif i == 1:
                    data = line.split()
                    a1 = float(data[10])
                    a2 = float(data[11])
                    a3 = float(data[12])
                    b1 = float(data[15])
                    b2 = float(data[16])
                    b3 = float(data[17])
                    c1 = float(data[20])
                    c2 = float(data[21])
                    c3 = float(data[22])

                elif i <= natoms + 1:
                    data = line.split()
                    symbols.append(data[0])
                    x = float(data[1])
                    y = float(data[2])
                    z = float(data[3])
                    charges.append(float(data[4]))
                    positions.append([x, y, z])
        cell = [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
        atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.info["_atom_site_charge"] = charges

        return atoms

    output_path = Path(graspa_input.output_path)
    name = graspa_input.name
    ddec6_path = graspa_input.ddec6_path
    adsorbate = graspa_input.adsorbate
    temperature = graspa_input.temperature
    pressure = graspa_input.pressure
    n_cycle = graspa_input.n_cycle
    cutoff = graspa_input.cutoff

    out_dir = output_path / f"{name}_{adsorbate}_{temperature}_{pressure:0e}"
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = _get_cif_from_chargemol(ddec6_path)
    # Write CIF file with charges
    _write_cif(atoms, out_dir=out_dir, name=name + ".cif")

    # Copy other input files (simulation.input, force fields and definition files) from template folder.
    subprocess.run(f"cp {_file_dir}/* {out_dir}/", shell=True)
    print(_file_dir)
    [uc_x, uc_y, uc_z] = _calculate_cell_size(atoms=atoms)

    # Modify input from template simulation.input
    with (
        open(f"{out_dir}/simulation.input", "r") as f_in,
        open(f"{out_dir}/simulation.input.tmp", "w") as f_out,
    ):
        for line in f_in:
            if "NCYCLE" in line:
                line = line.replace("NCYCLE", str(n_cycle))
            if "ADSORBATE" in line:
                line = line.replace("ADSORBATE", adsorbate)
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "PRESSURE" in line:
                line = line.replace("PRESSURE", str(pressure))
            if "UC_X UC_Y UC_Z" in line:
                line = line.replace("UC_X UC_Y UC_Z", f"{uc_x} {uc_y} {uc_z}")
            if "CUTOFF" in line:
                line = line.replace("CUTOFF", str(cutoff))
            if "CIFFILE" in line:
                line = line.replace("CIFFILE", name)
            f_out.write(line)

    shutil.move(f"{out_dir}/simulation.input.tmp", f"{out_dir}/simulation.input")

    # Run gRASPA-sycl
    subprocess.run(
        "mpiexec -n 1 /lus/flare/projects/MOFA/thang/gRASPA/graspa-sycl/bin/sycl.out >> raspa.log",
        shell=True,
        cwd=out_dir,
    )

    # Get output from Output/ folder
    with open(f"{out_dir}/raspa.log", "r") as rf:
        for line in rf:
            if "UnitCells" in line:
                unitcell_line = line.strip()
            elif "Overall: Average:" in line:
                uptake_line = line.strip()

    # gRASPA-sycl only output the total number of molecules
    # This section is for unit conversion.
    unitcell = unitcell_line.split()[4:]
    unitcell = [int(float(i)) for i in unitcell]
    uptake_total_molecule = float(uptake_line.split()[2][:-1])
    error_total_molecule = float(uptake_line.split()[4][:-1])

    # Get unit in mol/kg
    framework_mass = sum(atoms.get_masses())
    framework_mass = framework_mass * unitcell[0] * unitcell[1] * unitcell[2]
    uptake_mol_kg = uptake_total_molecule / framework_mass * 1000
    error_mol_kg = error_total_molecule / framework_mass * 1000

    # Get unit in g/L
    framework_vol = atoms.get_volume()  # in Angstrom^3
    framework_vol_in_L = framework_vol * 1e-27 * unitcell[0] * unitcell[1] * unitcell[2]

    # Hard code for CO2 and H2
    if adsorbate == "CO2":
        molar_mass = 44.0098
    elif adsorbate == "H2":
        molar_mass = 2.02
    else:
        raise ValueError(f"Adsorbate {adsorbate} is not supported.")
    uptake_g_L = uptake_total_molecule / (6.022 * 1e23) * molar_mass / framework_vol_in_L
    error_g_L = error_total_molecule / (6.022 * 1e23) * molar_mass / framework_vol_in_L
    return uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L
