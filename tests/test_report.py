import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from chemgraph.models.ase_input import ASEOutputSchema
from chemgraph.tools.report_tools import generate_html

# Sample ASE output data from the test file
sample_ase_output = {
    'converged': True,
    'final_structure': {
        'numbers': [6, 8, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'positions': [[-4.297338269684306, 0.10255573262949688, 0.19043461265772596], [-3.091168327666823, -0.5617892357920903, -0.09604678001427121], [-1.928129678407811, 0.025628038849525435, 0.2907183374663119], [-1.8022926255539486, 1.2286399413019007, 0.9666222725511413], [-0.551324959347487, 1.7448260322973186, 1.3172725447209122], [0.6193862082695265, 1.0924958583077207, 1.0153481041728794], [0.5228977173358975, -0.12374904162595737, 0.3371545449648246], [-0.7345351161802945, -0.6401088674531116, -0.01638980290839578], [-0.5960945565053299, -1.821091647245954, -0.6636490888895847], [0.7289033699287576, -2.0655132413059802, -0.7282035234804388], [1.4688971430663722, -1.0910006548801274, -0.13880705373310848], [2.9595558572778122, -1.012105749511873, -0.038379009310229084], [3.4535409161572113, -0.20226293131947976, -1.2179441474920607], [3.7076186086949936, -0.6421081758936452, -2.3049888825807843], [3.545795536065225, 1.0992748019508947, -0.9244992930629404], [-5.084490388713139, -0.5337027045849032, -0.21045159069183741], [-4.442841228070811, 0.22421715042636722, 1.2687388875532541], [-4.340131805235365, 1.0827330882046844, -0.29541847057870996], [-2.6808858273418346, 1.792388951429148, 1.2358153213759553], [-0.5189156923777047, 2.687944284425179, 1.8422595777634458], [1.5805439231383362, 1.5021673688938544, 1.280575256338717], [1.0298727728045178, -2.971365545193551, -1.2173334947858527], [3.389963926895973, -2.011444528744831, -0.09151821198339088], [3.246120239581823, -0.51372502385832, 0.8867083940652692], [3.815052255867611, 1.6070960986933644, -1.7064080932891852]],
        'cell': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        'pbc': [False, False, False]
    },
    'simulation_input': {
        'atomsdata': {'numbers': [6, 8, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'positions': [[-4.386389153880405, 0.08279395837817095, 0.17035293728963444], [-3.1272253754444743, -0.5293064652150616, -0.08652803796250216], [-1.8948390857194093, 0.019554729834529425, 0.294335450049076], [-1.7742033545538396, 1.2532251407598456, 0.9802125876017296], [-0.5018669285821701, 1.755680440673316, 1.3330537662335435], [0.663778948008566, 1.0386351545558137, 1.012033968722649], [0.5144687867255454, -0.17410036556653308, 0.34193491371306767], [-0.7168404827877023, -0.6633083772303519, -0.007817792913418472], [-0.5954553785556782, -1.806025041419148, -0.6419425737287208], [0.6921919396470129, -2.0897808883985323, -0.7239460416861158], [1.4392686899585787, -1.0926336456556074, -0.10598354427003526], [2.932367945009614, -1.0013229057877107, -0.03761338529227019], [3.447094638813016, -0.17613840419406154, -1.1719230253564632], [3.725478522448878, -0.720745665190186, -2.2736694091927494], [3.576070214541992, 1.2012692755231267, -1.024237917407537], [-5.192631614423245, -0.561698933243827, -0.23579528431768718], [-4.538915721231114, 0.1972289286135407, 1.2643716164952985], [-4.437591579097855, 1.0739105667618845, -0.32795039445661456], [-2.6504628973667756, 1.830115366842459, 1.2424797011718762], [-0.423255327155912, 2.7005824309182587, 1.8548281600138927], [1.6410436871288054, 1.419023767368517, 1.2791163438819162], [1.090387046393975, -2.9713177885918967, -1.2087345125638282], [3.3736953005678907, -2.020405878896555, -0.08229839088755009], [3.237383476570283, -0.5422986899155408, 0.9266661649231507], [3.9064477029836775, 1.777063289075177, -1.7893348892307053]], 'cell': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 'pbc': [False, False, False]},
        'driver': 'thermo',
        'optimizer': 'bfgs',
        'calculator': {'calculator_type': 'TBLite', 'method': 'GFN2-xTB'},
        'fmax': 0.01,
        'steps': 1000,
        'temperature': 800.0,
        'pressure': 101325.0
    },
    'single_point_energy': -1215.4074283058446,
    'energy_unit': 'eV',
    'vibrational_frequencies': {
        'energies': ['0.7074529691501289i', '0.3392448793636486i', '0.09984213460987901i', '0.0177440266539047', '0.1803511442414236', '0.3405341780536821', '2.124817531794364', '4.186157180258649', '5.993019051361853', '10.818356408664147', '17.62366607508873', '19.138500723439286', '21.80638835145705', '28.913903045713056', '29.152574687204147', '33.92465345310803', '41.75134316394184', '43.14879800796078', '48.94016626226598', '57.36235352348695', '59.55827653742017', '64.02799028573038', '68.23963212767663', '68.7017213971901', '72.42416051533078', '75.39433250357196', '79.26265669006733', '85.33730339639334', '86.40590174975839', '87.13860544787346', '95.51521570838649', '97.05572116388181', '98.20311401575842', '104.37258219915421', '106.26346793588695', '111.60948094263357', '112.96340313356049', '115.82399552362382', '116.57337966194258', '130.82044624178667', '134.64108895581472', '136.30982212962414', '138.03770875845643', '139.15631984941075', '143.39385072213074', '143.75683184888663', '145.1883769195418', '148.17332496004897', '149.86962379363624', '153.1462777127941', '154.17626504908574', '155.9508552307188', '158.5149266014786', '167.43220281671117', '170.35164753983256', '172.81639047066284', '173.75356565220824', '176.54915160031373', '182.1714310489433', '183.75305928516212', '186.7720602184985', '195.74719375833834', '196.59138481823015', '199.51601467365376', '222.95789696501413', '363.5003226038018', '365.248115026463', '373.13237702496497', '374.2677180148086', '375.74459786452485', '382.4931640939298', '385.2702547077921', '385.9338600553477', '392.6499391487497', '426.9911780341727'],
        'energy_unit': 'meV',
        'frequencies': ['5.705993054003747i', '2.7361945029091204i', '0.8052811302287161i', '0.14311522779933636', '1.4546300901944604', '2.7465933982289106', '17.13780930484885', '33.76363494836766', '48.336958880338074', '87.25592967364437', '142.14445425445348', '154.36241977071413', '175.8803848343414', '233.20635736559538', '235.13137399273862', '273.62078527348757', '336.74729554809954', '348.0185290872492', '394.7290645903247', '462.65858656233235', '480.36990026235435', '516.4205731881907', '550.389755796847', '554.1167571342518', '584.1402536467884', '608.0963065197592', '639.2964454707533', '688.291775794213', '696.910602836731', '702.8202567563546', '770.3821754235457', '782.8071899644711', '792.06153750207', '841.8216546214359', '857.0726767390943', '900.1911798952193', '911.1112989052098', '934.1835327070494', '940.2277234511552', '1055.1380658901421', '1085.9536278267374', '1099.4128686723598', '1113.3492143125047', '1122.3714212908492', '1156.5494130164527', '1159.4770532672128', '1171.0232430294304', '1195.0984728021697', '1208.7800456897514', '1235.2080420632055', '1243.5154502484481', '1257.8284854488877', '1278.5091159211881', '1350.4317996380644', '1373.9787095219685', '1393.8582020910892', '1401.417029758046', '1423.9649512416795', '1469.31169354186', '1482.0683856803103', '1506.418270558654', '1578.8076050731504', '1585.6164652151629', '1609.2051960118213', '1798.2767291880298', '2931.827847718401', '2945.9247444146226', '3009.5156065088795', '3018.6727492229115', '3030.5845886405937', '3085.0154465331375', '3107.414193084549', '3112.7665312170766', '3166.9353626845987', '3443.91613609418'],
        'frequency_unit': 'cm-1'
    },
    'thermochemistry': {
        'enthalpy': -1215.4074283058446,
        'entropy': 0.008507643747397382,
        'gibbs_free_energy': -1222.2135433037624,
        'unit': 'eV'
    },
    'success': True,
    'error': ''
}

def create_xyz_content_from_final_structure(final_structure):
    """Create XYZ file content from a final_structure dictionary."""
    num_atoms = len(final_structure['numbers'])
    xyz_lines = [str(num_atoms), "Sample molecule"]
    element_map = {1: 'H', 6: 'C', 8: 'O'}  # Simplified element map for this example

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
            "xyz_path": str(xyz_path)
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
        assert "-1215.407428" in html_content  # Check for the value separately
        assert "Entropy" in html_content
        assert "Gibbs Free Energy" in html_content
        assert "Thermochemistry Values" in html_content
        assert "Energy Unit" in html_content
        assert "eV" in html_content
        assert "kJ/mol" in html_content
        assert "kcal/mol" in html_content
        
        # Check for vibrational frequencies
        assert "Vibrational Frequencies" in html_content  # Check for the label
        assert "5.705993054003747i" in html_content  # Check for first frequency
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
            "ase_output": sample_ase_output_schema
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
        assert "-1215.407428" in html_content  # Check for the value separately
        assert "Entropy" in html_content
        assert "Gibbs Free Energy" in html_content
        assert "Thermochemistry Values" in html_content
        assert "Energy Unit" in html_content
        assert "eV" in html_content
        assert "kJ/mol" in html_content
        assert "kcal/mol" in html_content
        
        # Check for vibrational frequencies
        assert "Vibrational Frequencies" in html_content  # Check for the label
        assert "5.705993054003747i" in html_content  # Check for first frequency
        assert "cm-1" in html_content  # Check for unit
        
        # Verify the NGL viewer is included
        assert "ngl.js" in html_content
        assert "new NGL.Stage" in html_content
        
        # Verify the XYZ content is properly encoded
        assert "const xyzData = atob(" in html_content
        assert "stage.loadFile" in html_content


