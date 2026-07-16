import os
import pytest
from ase.build import bulk
from ase.io import write
from chemgraph.tools.phonopy_tools import run_phonopy
from chemgraph.schemas.phonopy_schema import PhonopyInputSchema

def test_run_phonopy_emt(tmp_path):
    # Create a simple Cu FCC structure
    atoms = bulk("Cu", "fcc", a=3.6)
    structure_file = str(tmp_path / "POSCAR_Cu")
    write(structure_file, atoms, format="vasp")

    output_json = str(tmp_path / "phonopy_results.json")
    
    # Input schema for phonopy
    params = PhonopyInputSchema(
        input_structure_file=structure_file,
        output_results_file=output_json,
        supercell_matrix=[2, 2, 2],
        calculator={
            "calculator_type": "EMT",
        },
        calculate_dos=True,
        calculate_thermal_properties=True,
        calculate_band_structure=True,
        band_npoints=5,
    )
    
    result = run_phonopy.invoke({"params": params})
    
    assert result["status"] == "success"
    assert os.path.exists(output_json)
    
    # Verify .dat file exists
    assert result["band_dat"] is not None
    assert os.path.exists(result["band_dat"])
    
    # Verify .yaml file exists
    assert result["band_yaml"] is not None
    assert os.path.exists(result["band_yaml"])
    
    # Check content of .dat file
    with open(result["band_dat"]) as f:
        content = f.read()
        assert "End points of segments" in content
        
    # Verify report file exists and is in English
    assert result["calculation_info_file"] is not None
    assert os.path.exists(result["calculation_info_file"])
    with open(result["calculation_info_file"]) as f_info:
        info_content = f_info.read()
        assert "# Phonon Calculation Information" in info_content
        assert "Calculator & Model Configuration" in info_content
        assert "Structure Minimization" in info_content
        assert "Supercell Configuration" in info_content
        assert "Reciprocal Space" in info_content
        assert "Generated Output Files" in info_content
        assert "Minimized Structure (VASP)" in info_content
        assert "Minimization Log" in info_content

    # Verify minimized structure file and log exist and are returned
    assert result["minimized_structure_file"] is not None
    assert os.path.exists(result["minimized_structure_file"])
    assert result["minimization_log_file"] is not None
    assert os.path.exists(result["minimization_log_file"])


def test_run_phonopy_relaxation_modes(tmp_path):
    import numpy as np
    from ase.io import read
    
    # Test "atoms" mode (only atoms change, cell stays fixed)
    atoms = bulk("Cu", "fcc", a=3.6) * [2, 1, 1]
    orig_lengths = list(atoms.cell.lengths())
    atoms.positions[0] += [0.1, 0.0, 0.0]
    structure_file = str(tmp_path / "POSCAR_Cu_atoms")
    write(structure_file, atoms, format="vasp")
    output_json = str(tmp_path / "results_atoms.json")
    
    params = PhonopyInputSchema(
        input_structure_file=structure_file,
        output_results_file=output_json,
        supercell_matrix=[2, 2, 2],
        relaxation_mode="atoms",
        calculator={"calculator_type": "EMT"},
        calculate_dos=False,
        calculate_thermal_properties=False,
        calculate_band_structure=False,
    )
    result = run_phonopy.invoke({"params": params})
    assert result["status"] == "success"
    
    # Read minimized structure
    min_atoms = read(result["minimized_structure_file"])
    # Lattice lengths should be identical to starting lengths
    assert np.allclose(min_atoms.cell.lengths(), orig_lengths)
    # Positions should have shifted (optimized)
    assert not np.allclose(min_atoms.positions[0], atoms.positions[0])



