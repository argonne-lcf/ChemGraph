import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from chemgraph.models.ase_input import ASEOutputSchema
from chemgraph.tools.report_tools import generate_html

TEST_DIR = Path(__file__).parent

# Sample ASE output data from the test file
sample_ase_output = {
    "input_structure_file": str(TEST_DIR / "water.xyz"),
    "converged": True,
    "final_structure": {
        "numbers": [8, 1, 1],
        "positions": [
            [0.0, -1.497182414522372e-18, 0.39869538657185805],
            [5.691224674128633e-20, 0.7641487689241104, -0.19934769828592902],
            [-6.545249382012399e-22, -0.7641487689241104, -0.19934769828592905],
        ],
        "cell": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "pbc": [False, False, False],
    },
    "simulation_input": {
        "input_structure_file": "water.xyz",
        "output_results_file": "water_output.json",
        "driver": "thermo",
        "optimizer": "bfgs",
        "calculator": {
            "calculator_type": "mace_mp",
            "model": None,
            "device": "cpu",
            "default_dtype": "float64",
            "dispersion": False,
            "damping": "bj",
            "dispersion_xc": "pbe",
            "dispersion_cutoff": 21.167088422553647,
        },
        "fmax": 0.01,
        "steps": 1000,
        "temperature": 298.0,
        "pressure": 101325.0,
    },
    "single_point_energy": -13.786080327355158,
    "energy_unit": "eV",
    "dipole_value": [None, None, None],
    "dipole_unit": " e * angstrom",
    "vibrational_frequencies": {
        "energies": [
            "6.9742290358871735i",
            "3.2710270769584744i",
            "0.04848856978180061i",
            "0.005305115127349197i",
            "0.04569285207959072",
            "5.23408996896935",
            "207.54494811634646",
            "461.81541979826835",
            "484.1391752707912",
        ],
        "energy_unit": "meV",
        "frequencies": [
            "56.250951188471966i",
            "26.382612830086863i",
            "0.3910866933047204i",
            "0.042788639510142355i",
            "0.3685377091525602",
            "42.21578296978211",
            "1673.962912005435",
            "3724.792590476138",
            "3904.84582259207",
        ],
        "frequency_unit": "cm-1",
    },
    "ir_data": {},
    "thermochemistry": {
        "enthalpy": -13.10654781092989,
        "entropy": 0.001957587821789186,
        "gibbs_free_energy": -13.689908981823066,
        "unit": "eV",
    },
    "success": True,
    "error": "",
    "wall_time": 0.7555508613586426,
}


def create_xyz_content_from_final_structure(final_structure):
    """Create XYZ file content from a final_structure dictionary."""
    num_atoms = len(final_structure['numbers'])
    xyz_lines = [str(num_atoms), "Sample molecule"]
    element_map = {1: 'H', 8: 'O'}  # Simplified element map for this example

    for num, pos in zip(final_structure['numbers'], final_structure['positions']):
        element = element_map.get(num, f"X{num}")  # Use X{num} for unknown elements
        x, y, z = pos
        xyz_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")

    return "\n".join(xyz_lines)


@pytest.fixture
def sample_ase_output_schema():
    """Create a valid ASEOutputSchema object from the sample data."""
    return ASEOutputSchema(**sample_ase_output)


@pytest.fixture(scope="session")
def test_output_dir():
    """Create a test output directory for saving HTML files for inspection."""
    # Create a test_outputs directory in the tests folder
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    # Create a timestamped subdirectory for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    yield run_dir

    # Optionally clean up old test outputs (keep last 5 runs)
    if output_dir.exists():
        runs = sorted(output_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        for old_run in runs[5:]:  # Keep only the 5 most recent runs
            shutil.rmtree(old_run)


def test_generate_html_with_xyz(test_output_dir, sample_ase_output_schema):
    """Test the generate_html function with an external XYZ file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input XYZ file
        xyz_path = Path(tmpdir) / "molecule.xyz"
        xyz_content = create_xyz_content_from_final_structure(sample_ase_output['final_structure'])
        with open(xyz_path, "w") as f:
            f.write(xyz_content)

        # Create output HTML file path
        output_path = Path(tmpdir) / "report.html"

        # Generate HTML report with xyz_path
        result = generate_html.invoke({
            "output_path": str(output_path),
            "ase_output": sample_ase_output_schema,
            "xyz_path": str(xyz_path),
        })

        # Verify the returned path matches our output path
        assert result == str(output_path)

        # Verify the HTML file was created
        assert output_path.exists()

        # Save a copy of the HTML file for inspection
        inspection_path = test_output_dir / "report_with_xyz.html"
        shutil.copy2(output_path, inspection_path)
        print(f"\nSaved HTML report with XYZ file to: {inspection_path}")

        # Read the generated HTML content
        with open(output_path, "r") as f:
            html_content = f.read()

        # Verify key information is present in the HTML
        assert "XYZ Molecule Viewer" in html_content
        assert "Calculation Results" in html_content
        assert "Simulation Details" in html_content

        # Check for collapsible functionality
        assert "toggleSection" in html_content
        assert "collapsible-content" in html_content
        assert "onclick=\"toggleSection" in html_content

        # Check for simulation details
        assert "Simulation Type" in html_content
        assert "Calculator" in html_content
        assert "thermo" in html_content  # From sample data
        assert "Temperature" in html_content
        assert "Pressure" in html_content

        # Check for thermochemistry values
        assert "Enthalpy" in html_content
        assert "Entropy" in html_content
        assert "Gibbs Free Energy" in html_content
        assert "Thermochemistry Values" in html_content
        assert "Energy Unit" in html_content
        assert "eV" in html_content
        assert "kJ/mol" in html_content
        assert "kcal/mol" in html_content

        # Check for vibrational frequencies
        assert "Vibrational Frequencies" in html_content  # Check for the label
        assert "cm-1" in html_content  # Check for unit

        # Verify the NGL viewer is included
        assert "ngl.js" in html_content
        assert "new NGL.Stage" in html_content

        # Verify the XYZ content is properly encoded
        assert "const xyzData = atob(" in html_content
        assert "stage.loadFile" in html_content


def test_generate_html_without_xyz(test_output_dir, sample_ase_output_schema):
    """Test the generate_html function using only the final_structure from ASE output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create output HTML file path
        output_path = Path(tmpdir) / "report.html"

        # Generate HTML report without xyz_path
        result = generate_html.invoke({
            "output_path": str(output_path),
            "ase_output": sample_ase_output_schema,
        })

        # Verify the returned path matches our output path
        assert result == str(output_path)

        # Verify the HTML file was created
        assert output_path.exists()

        # Save a copy of the HTML file for inspection
        inspection_path = test_output_dir / "report_without_xyz.html"
        shutil.copy2(output_path, inspection_path)
        print(f"\nSaved HTML report without XYZ file to: {inspection_path}")

        # Read the generated HTML content
        with open(output_path, "r") as f:
            html_content = f.read()

        # Verify key information is present in the HTML
        assert "XYZ Molecule Viewer" in html_content
        assert "Calculation Results" in html_content
        assert "Simulation Details" in html_content

        # Check for collapsible functionality
        assert "toggleSection" in html_content
        assert "collapsible-content" in html_content
        assert "onclick=\"toggleSection" in html_content

        # Check for simulation details
        assert "Simulation Type" in html_content
        assert "Calculator" in html_content
        assert "thermo" in html_content  # From sample data
        assert "Temperature" in html_content
        assert "Pressure" in html_content

        # Check for thermochemistry values
        assert "Enthalpy" in html_content
        assert "Entropy" in html_content
        assert "Gibbs Free Energy" in html_content
        assert "Thermochemistry Values" in html_content
        assert "Energy Unit" in html_content
        assert "eV" in html_content
        assert "kJ/mol" in html_content
        assert "kcal/mol" in html_content

        # Check for vibrational frequencies
        assert "Vibrational Frequencies" in html_content  # Check for the label
        assert "cm-1" in html_content  # Check for unit

        # Verify the NGL viewer is included
        assert "ngl.js" in html_content
        assert "new NGL.Stage" in html_content

        # Verify the XYZ content is properly encoded
        assert "const xyzData = atob(" in html_content
        assert "stage.loadFile" in html_content
