import base64
import sys
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from chemgraph.models.ase_input import ASEOutputSchema

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>XYZ Molecule Viewer</title>
    <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 1.5rem;
            font-weight: 600;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-weight: 500;
        }}
        #viewer {{
            width: 100%;
            height: 600px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background: white;
            margin: 1rem 0;
        }}
        .info-section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 2rem;
        }}
        .info-section ul {{
            list-style-type: none;
            padding: 0;
            margin: 0;
        }}
        .info-section li {{
            padding: 0.75rem 0;
            border-bottom: 1px solid #eee;
        }}
        .info-section li:last-child {{
            border-bottom: none;
        }}
        pre {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            margin: 0.5rem 0;
        }}
        code {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>XYZ Molecule Viewer</h1>
        <div id="viewer"></div>
        <div class="info-section">
            <h2>Calculation Results</h2>
            <ul id="calculation-results">
                <!-- Results will be populated here -->
            </ul>
        </div>
    </div>
    <script>
        const stage = new NGL.Stage("viewer", {{ backgroundColor: "white" }});

        function xyzToPDB(xyzContent) {{
            const lines = xyzContent.trim().split("\\n");
            const numAtoms = parseInt(lines[0]);
            let pdbContent = '';
            for (let i = 2; i < lines.length && i < numAtoms + 2; i++) {{
                const parts = lines[i].trim().split(/\\s+/);
                if (parts.length >= 4) {{
                    const [atom, x, y, z] = parts;
                    const atomName = atom.padEnd(3);
                    const serial = String(i - 1).padStart(5);
                    const xStr = parseFloat(x).toFixed(3).padStart(8);
                    const yStr = parseFloat(y).toFixed(3).padStart(8);
                    const zStr = parseFloat(z).toFixed(3).padStart(8);
                    pdbContent += `HETATM${{serial}} ${{atomName}} MOL     1    ${{xStr}}${{yStr}}${{zStr}}  1.00  0.00\\n`;
                }}
            }}
            pdbContent += 'END\\n';
            return pdbContent;
        }}

        const xyzData = atob("{encoded_xyz}");

        const pdbContent = xyzToPDB(xyzData);
        stage.loadFile(new Blob([pdbContent], {{ type: 'text/plain' }}), {{ ext: 'pdb' }}).then(component => {{
            component.addRepresentation("ball+stick");
            component.autoView();
        }});
    </script>
</body>
</html>
"""

@tool
def generate_html(output_path: Path, ase_output: ASEOutputSchema, xyz_path: Optional[Path] = None) -> str:
    """Generate an HTML report from ASE output, optionally using an XYZ file for visualization.
    
    Parameters
    ----------
    output_path : Path
        Path where the HTML report will be saved
    ase_output : ASEOutputSchema
        The output from an ASE calculation containing energy, frequencies, etc.
    xyz_path : Optional[Path]
        Optional path to an XYZ file. If not provided, the final_structure from ase_output will be used.
        
    Returns
    -------
    str
        Path to the generated HTML file
    """
    # Get XYZ content either from file or final_structure
    if xyz_path is not None:
        with open(xyz_path, 'r') as f:
            xyz_content = f.read()
    else:
        # Convert final_structure to XYZ format
        num_atoms = len(ase_output.final_structure.numbers)
        xyz_lines = [str(num_atoms), "Optimized Structure"]
        
        # Map atomic numbers to element symbols
        element_map = {
            1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
            11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
            # Add more elements as needed
        }
        
        for num, pos in zip(ase_output.final_structure.numbers, ase_output.final_structure.positions):
            element = element_map.get(num, f"X{num}")  # Use X{num} for unknown elements
            x, y, z = pos
            xyz_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
        
        xyz_content = "\n".join(xyz_lines)

    encoded_xyz = base64.b64encode(xyz_content.encode()).decode()
    html_content = HTML_TEMPLATE.format(encoded_xyz=encoded_xyz)
    
    # Add additional information to the HTML content
    html_content = add_additional_info_to_html(html_content, ase_output)

    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"✅ HTML viewer created: {output_path}")
    return str(output_path)

def add_additional_info_to_html(html_content: str, ase_output: ASEOutputSchema) -> str:
    """Add ASE calculation results to the HTML content.
    
    Parameters
    ----------
    html_content : str
        The base HTML content
    ase_output : ASEOutputSchema
        The output from an ASE calculation
        
    Returns
    -------
    str
        HTML content with additional information added
    """
    additional_info = ""
    
    # Optimized Coordinates (from final structure)
    if ase_output.final_structure is not None:
        # Convert AtomsData to XYZ format
        num_atoms = len(ase_output.final_structure.numbers)
        xyz_lines = [str(num_atoms), "Optimized Structure"]
        
        # Map atomic numbers to element symbols
        element_map = {
            1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
            11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
            # Add more elements as needed
        }
        
        for num, pos in zip(ase_output.final_structure.numbers, ase_output.final_structure.positions):
            element = element_map.get(num, f"X{num}")  # Use X{num} for unknown elements
            x, y, z = pos
            xyz_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
        
        xyz_str = "\n".join(xyz_lines)
        additional_info += f"<li><strong>Optimized Coordinates:</strong><pre><code>{xyz_str}</code></pre></li>"
    else:
        additional_info += "<li><strong>Optimized Coordinates:</strong> N/A</li>"
    
    # Energy
    if ase_output.single_point_energy is not None:
        additional_info += f"<li><strong>Single Point Energy</strong> ({ase_output.energy_unit}): {ase_output.single_point_energy:.6f}</li>"
    else:
        additional_info += f"<li><strong>Single Point Energy</strong> ({ase_output.energy_unit}): N/A</li>"
    
    # Vibrational Frequencies
    if ase_output.vibrational_frequencies and "frequencies" in ase_output.vibrational_frequencies:
        freq_str = ", ".join(ase_output.vibrational_frequencies["frequencies"])
        unit = ase_output.vibrational_frequencies.get("frequency_unit", "cm-1")
        additional_info += f"<li><strong>Vibrational Frequencies</strong> ({unit}): {freq_str}</li>"
    else:
        additional_info += "<li><strong>Vibrational Frequencies:</strong> N/A</li>"
    
    # Thermochemistry Values
    if ase_output.thermochemistry:
        thermo_info = []
        if "enthalpy" in ase_output.thermochemistry:
            thermo_info.append(f"Enthalpy: {ase_output.thermochemistry['enthalpy']:.6f}")
        if "entropy" in ase_output.thermochemistry:
            thermo_info.append(f"Entropy: {ase_output.thermochemistry['entropy']:.6f}")
        if "gibbs_free_energy" in ase_output.thermochemistry:
            thermo_info.append(f"Gibbs Free Energy: {ase_output.thermochemistry['gibbs_free_energy']:.6f}")
        if thermo_info:
            unit = ase_output.thermochemistry.get("unit", "eV")
            additional_info += f"<li><strong>Thermochemistry Values</strong> ({unit}):<br>" + "<br>".join(thermo_info) + "</li>"
        else:
            additional_info += "<li><strong>Thermochemistry Values:</strong> N/A</li>"
    else:
        additional_info += "<li><strong>Thermochemistry Values:</strong> N/A</li>"
    
    # Optimization Status
    if ase_output.simulation_input.driver == "opt":
        status = "Converged" if ase_output.converged else "Not Converged"
        status_class = "color: #28a745;" if ase_output.converged else "color: #dc3545;"
        additional_info += f"<li><strong>Optimization Status:</strong> <span style='{status_class}'>{status}</span></li>"
    
    # Error Information
    if ase_output.error:
        additional_info += f"<li><strong>Error:</strong> <span style='color: #dc3545;'>{ase_output.error}</span></li>"
    
    # Replace the empty ul with our populated content
    html_content = html_content.replace('<ul id="calculation-results">\n                <!-- Results will be populated here -->\n            </ul>', f'<ul id="calculation-results">{additional_info}</ul>')
    return html_content

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_xyz_viewer.py molecule.xyz output.html")
        sys.exit(1)

    xyz_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    # Note: This will now require an ASEOutputSchema object
    print("Error: This script now requires an ASEOutputSchema object. Please use the generate_html function directly.")
    sys.exit(1)
