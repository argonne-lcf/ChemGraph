import base64
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from chemgraph.models.ase_input import ASEOutputSchema
from chemgraph.tools.ase_tools import is_linear_molecule


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
            display: flex;
            align-items: center;
            cursor: pointer;
        }}
        h2::before {{
            content: '▼';
            font-size: 0.8em;
            margin-right: 0.5rem;
            transition: transform 0.3s ease;
        }}
        h2.collapsed::before {{
            transform: rotate(-90deg);
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
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .info-section li {{
            padding: 0;
            border: none;
            margin: 0;
        }}
        .sub-section {{
            border: 1px solid #eee;
            border-radius: 4px;
            padding: 0.5rem;
            background: #fafafa;
        }}
        .sub-section h3 {{
            color: #2c3e50;
            margin: 0;
            padding: 0.5rem;
            font-size: 1.1em;
            font-weight: 500;
            display: flex;
            align-items: center;
            cursor: pointer;
            user-select: none;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .sub-section h3::before {{
            content: '▼';
            font-size: 0.8em;
            margin-right: 0.5rem;
            transition: transform 0.3s ease;
            display: inline-block;
        }}
        .sub-section h3.collapsed::before {{
            transform: rotate(-90deg);
        }}
        .sub-section-content {{
            display: block;
            padding: 0.5rem;
            margin-top: 0.5rem;
        }}
        .sub-section-content.collapsed {{
            display: none;
        }}
        .table-container {{
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #eee;
            border-radius: 4px;
            margin: 0.5rem 0;
        }}
        .table-container table {{
            margin: 0;
            border-collapse: collapse;
            width: 100%;
        }}
        .table-container thead {{
            position: sticky;
            top: 0;
            background: #f8f9fa;
            z-index: 1;
        }}
        .table-container th {{
            border-bottom: 2px solid #dee2e6;
        }}
        .table-container tbody tr:last-child td {{
            border-bottom: none;
        }}
        .unit-toggle {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .unit-toggle button {{
            padding: 0.25rem 0.5rem;
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .unit-toggle button.active {{
            background: #2c3e50;
            color: white;
            border-color: #2c3e50;
        }}
        .unit-toggle button:hover:not(.active) {{
            background: #e9ecef;
        }}
        .energy-value {{
            display: inline-block;
            min-width: 100px;
        }}
        .regular-item {{
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #eee;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0.5rem 0;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{
            background-color: #f8f9fa;
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
        .collapsible-content {{
            transition: max-height 0.3s ease-out;
            max-height: 2000px;
            overflow: hidden;
        }}
        .collapsible-content.collapsed {{
            max-height: 0;
        }}
        .vibrational-mode {{
            background-color: #e8f5e9;
        }}
        .trans-rot-mode {{
            background-color: #fff3e0;
        }}
        .mode-explanation {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
        }}
        .mode-explanation p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>XYZ Molecule Viewer</h1>
        <div id="viewer"></div>
        <div class="info-section">
            <h2 onclick="toggleSection('calculation-results')">Calculation Results</h2>
            <div id="calculation-results" class="collapsible-content">
                <ul>
                    <!-- Results will be populated here -->
                </ul>
            </div>
        </div>
        <div class="info-section">
            <h2 onclick="toggleSection('simulation-details')">Simulation Details</h2>
            <div id="simulation-details" class="collapsible-content">
                <ul>
                    <!-- Simulation details will be populated here -->
                </ul>
            </div>
        </div>
    </div>
    <script>
        function toggleSection(sectionId) {{
            const content = document.getElementById(sectionId);
            const header = content.previousElementSibling;
            content.classList.toggle('collapsed');
            header.classList.toggle('collapsed');
        }}

        function toggleSubSection(sectionId) {{
            const content = document.getElementById(sectionId);
            const header = content.previousElementSibling;
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                header.classList.remove('collapsed');
            }} else {{
                content.style.display = 'none';
                header.classList.add('collapsed');
            }}
        }}

        // Initialize all sub-sections as expanded
        document.addEventListener('DOMContentLoaded', function() {{
            const subSections = document.querySelectorAll('.sub-section-content');
            subSections.forEach(section => {{
                section.style.display = 'block';
            }});
        }});

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

    with open(output_path, 'w', encoding='utf-8') as f:
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
    # Calculation Results section
    calc_results = []
    
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
        calc_results.append(f"""
        <li>
            <div class="sub-section">
                <h3 onclick="toggleSubSection('optimized-coords')">Optimized Coordinates</h3>
                <div id="optimized-coords" class="sub-section-content">
                    <pre><code>{xyz_str}</code></pre>
                </div>
            </div>
        </li>""")
    else:
        calc_results.append("<li class='regular-item'><strong>Optimized Coordinates:</strong> N/A</li>")
    
    # Energy
    if ase_output.single_point_energy is not None:
        energy_ev = ase_output.single_point_energy
        calc_results.append(f"""
        <li class='regular-item'>
            <div class="unit-toggle">
                <span>Energy Unit:</span>
                <button onclick="toggleEnergyUnit('ev')" class="active" data-unit="ev">eV</button>
                <button onclick="toggleEnergyUnit('kjmol')" data-unit="kjmol">kJ/mol</button>
                <button onclick="toggleEnergyUnit('kcalmol')" data-unit="kcalmol">kcal/mol</button>
            </div>
            <div>
                <strong>Single Point Energy</strong> (<span class="energy-unit">eV</span>): 
                <span class="energy-value" data-ev="{energy_ev:.6f}"></span>
            </div>
        </li>""")
    else:
        calc_results.append(f"<li class='regular-item'><strong>Single Point Energy</strong> ({ase_output.energy_unit}): N/A</li>")
    
    # Vibrational Frequencies
    if ase_output.vibrational_frequencies and "frequencies" in ase_output.vibrational_frequencies:
        freq_unit = ase_output.vibrational_frequencies.get("frequency_unit", "cm-1")
        energy_unit = ase_output.vibrational_frequencies.get("energy_unit", "meV")
        
        # Check if molecule is linear
        is_linear = is_linear_molecule.invoke({"atomsdata": ase_output.final_structure})
        num_atoms = len(ase_output.final_structure.numbers)
        trans_rot_modes = 5 if is_linear else 6  # Number of translation/rotation modes
        
        # Create table header
        freq_table = f"""
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th style="width: 80px;">Mode #</th>
                        <th>Frequency ({freq_unit})</th>
                        <th>Energy ({energy_unit})</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add table rows with mode numbers and highlighting
        for i, (freq, energy) in enumerate(zip(
            ase_output.vibrational_frequencies["frequencies"],
            ase_output.vibrational_frequencies["energies"]
        ), 1):
            # First 5 (linear) or 6 (non-linear) modes are translation/rotation
            is_vibrational = i > trans_rot_modes
            mode_type = "Translation/Rotation" if i <= trans_rot_modes else "Vibrational"
            row_class = "trans-rot-mode" if i <= trans_rot_modes else "vibrational-mode"
            
            freq_table += f"""
                    <tr class="{row_class}">
                        <td>{i}</td>
                        <td>{freq}</td>
                        <td>{energy}</td>
                        <td>{mode_type}</td>
                    </tr>
            """
        
        freq_table += """
                </tbody>
            </table>
        </div>
        """
        
        # Add explanation about the modes
        mode_explanation = f"""
        <div class="mode-explanation">
            <p><strong>Molecule Type:</strong> {'Linear' if is_linear else 'Non-linear'}</p>
            <p><strong>Mode Breakdown:</strong> {trans_rot_modes} translation/rotation modes + {3 * num_atoms - trans_rot_modes} vibrational modes</p>
            <p><em>Note: The first {trans_rot_modes} modes (highlighted in orange) are translation/rotation modes. The remaining modes (highlighted in green) are vibrational modes.</em></p>
        </div>
        """
        
        calc_results.append(f"""
        <li>
            <div class="sub-section">
                <h3 onclick="toggleSubSection('vib-freqs')">Vibrational Frequencies</h3>
                <div id="vib-freqs" class="sub-section-content">
                    {mode_explanation}
                    {freq_table}
                </div>
            </div>
        </li>""")
    else:
        calc_results.append("<li class='regular-item'><strong>Vibrational Frequencies:</strong> N/A</li>")
    
    # Thermochemistry Values
    if ase_output.thermochemistry:
        thermo_info = []
        unit = ase_output.thermochemistry.get("unit", "eV")
        
        # Add data attributes for conversion with labels
        if "enthalpy" in ase_output.thermochemistry:
            enthalpy_ev = ase_output.thermochemistry['enthalpy']
            thermo_info.append(f'<div><strong>Enthalpy:</strong> <span class="energy-value" data-ev="{enthalpy_ev}">{enthalpy_ev:.6f}</span></div>')
        if "entropy" in ase_output.thermochemistry:
            entropy_ev = ase_output.thermochemistry['entropy']
            thermo_info.append(f'<div><strong>Entropy:</strong> <span class="energy-value" data-ev="{entropy_ev}">{entropy_ev:.6f}</span></div>')
        if "gibbs_free_energy" in ase_output.thermochemistry:
            gibbs_ev = ase_output.thermochemistry['gibbs_free_energy']
            thermo_info.append(f'<div><strong>Gibbs Free Energy:</strong> <span class="energy-value" data-ev="{gibbs_ev}">{gibbs_ev:.6f}</span></div>')
        
        if thermo_info:
            calc_results.append(f"""
            <li class='regular-item'>
                <div class="unit-toggle">
                    <span>Energy Unit:</span>
                    <button onclick="toggleEnergyUnit('ev')" class="active" data-unit="ev">eV</button>
                    <button onclick="toggleEnergyUnit('kjmol')" data-unit="kjmol">kJ/mol</button>
                    <button onclick="toggleEnergyUnit('kcalmol')" data-unit="kcalmol">kcal/mol</button>
                </div>
                <strong>Thermochemistry Values</strong> (<span class="energy-unit">eV</span>):<br>
                {"".join(thermo_info)}
            </li>""")
        else:
            calc_results.append("<li class='regular-item'><strong>Thermochemistry Values:</strong> No values available</li>")
    else:
        calc_results.append("<li class='regular-item'><strong>Thermochemistry Values:</strong> N/A</li>")
    
    # Optimization Status
    if ase_output.simulation_input.driver == "opt":
        status = "Converged" if ase_output.converged else "Not Converged"
        status_class = "color: #28a745;" if ase_output.converged else "color: #dc3545;"
        calc_results.append(f"<li class='regular-item'><strong>Optimization Status:</strong> <span style='{status_class}'>{status}</span></li>")
    
    # Error Information
    if ase_output.error:
        calc_results.append(f"<li class='regular-item'><strong>Error:</strong> <span style='color: #dc3545;'>{ase_output.error}</span></li>")

    # Join all results with proper spacing
    calc_results_html = "\n".join(calc_results)
    
    # Simulation Details section
    sim_details = []
    
    # Driver and Calculator
    sim_details.append(f"<li><strong>Simulation Type:</strong> {ase_output.simulation_input.driver or 'N/A'}</li>")
    
    if ase_output.simulation_input.calculator:
        calc = ase_output.simulation_input.calculator
        calc_type = calc.calculator_type
        sim_details.append(f"<li><strong>Calculator:</strong> {calc_type}</li>")
        
        # Get calculator parameters directly from the input
        calc_params = calc.model_dump()

        # Create a sub-section for calculator parameters
        calc_params_html = []
        for param, value in calc_params.items():
            # Format the parameter name nicely
            param_name = param.replace('_', ' ').title()
            
            # Handle boolean values
            if isinstance(value, bool):
                value = str(value)
            # Handle numeric values
            elif isinstance(value, (int, float)):
                value = f"{value:.6g}"
            # Handle None values
            elif value is None:
                value = "None"
            
            calc_params_html.append(f"<tr><td>{param_name}</td><td>{value}</td></tr>")

        # If no parameters are set, show an empty table
        if not calc_params_html:
            calc_params_html.append("<tr><td colspan='2'>No additional parameters set</td></tr>")
        
        sim_details.append(f"""
        <li>
            <div class='sub-section'>
                <h3 onclick='toggleSubSection("calc-params")'>Calculator Parameters</h3>
                <div id='calc-params' class='sub-section-content'>
                    <div class='table-container' style='max-height: 200px; overflow-y: auto;'>
                        <table>
                            <thead>
                                <tr>
                                    <th>Parameter</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(calc_params_html)}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </li>""")

    # Optimization Parameters
    if ase_output.simulation_input.driver == "opt":
        sim_details.append(f"<li><strong>Optimizer:</strong> {ase_output.simulation_input.optimizer}</li>")
        sim_details.append(f"<li><strong>Force Convergence (fmax):</strong> {ase_output.simulation_input.fmax} eV/Å</li>")
        sim_details.append(f"<li><strong>Maximum Steps:</strong> {ase_output.simulation_input.steps}</li>")
    
    # Thermochemistry Parameters
    if ase_output.simulation_input.driver == "thermo":
        if ase_output.simulation_input.temperature:
            sim_details.append(f"<li><strong>Temperature:</strong> {ase_output.simulation_input.temperature} K</li>")
        sim_details.append(f"<li><strong>Pressure:</strong> {ase_output.simulation_input.pressure} Pa</li>")
    
    # Join all simulation details
    sim_details_html = "\n".join(sim_details)
    
    # Replace the empty content in both sections
    html_content = html_content.replace(
        '<ul>\n                    <!-- Results will be populated here -->\n                </ul>',
        f'<ul>{calc_results_html}</ul>',
        1
    )
    html_content = html_content.replace(
        '<ul>\n                    <!-- Simulation details will be populated here -->\n                </ul>',
        f'<ul>{sim_details_html}</ul>',
        1
    )
    
    # Add the JavaScript for unit conversion
    html_content = html_content.replace(
        '</body>',
        '''
        <script>
            // Conversion factors
            const EV_TO_KJMOL = 96.485;  // 1 eV = 96.485 kJ/mol
            const EV_TO_KCALMOL = 23.061;  // 1 eV = 23.061 kcal/mol
            
            function toggleEnergyUnit(unit) {
                // Update button states
                document.querySelectorAll('.unit-toggle button').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.unit === unit);
                });
                
                // Update unit labels
                document.querySelectorAll('.energy-unit').forEach(label => {
                    label.textContent = unit === 'ev' ? 'eV' : 
                                     unit === 'kjmol' ? 'kJ/mol' : 'kcal/mol';
                });
                
                // Convert all energy values
                document.querySelectorAll('.energy-value').forEach(cell => {
                    const evValue = parseFloat(cell.dataset.ev);
                    let convertedValue;
                    let precision;
                    
                    if (unit === 'ev') {
                        convertedValue = evValue;
                        precision = 6;
                    } else if (unit === 'kjmol') {
                        convertedValue = evValue * EV_TO_KJMOL;
                        precision = 2;
                    } else {  // kcal/mol
                        convertedValue = evValue * EV_TO_KCALMOL;
                        precision = 2;
                    }
                    
                    // Get the label from the parent div's strong element
                    const label = cell.parentElement.querySelector('strong').textContent;
                    // Update only the value part
                    cell.textContent = convertedValue.toFixed(precision);
                });
            }

            // Initialize energy values in eV when the page loads
            document.addEventListener('DOMContentLoaded', function() {
                // Set initial unit to eV
                document.querySelectorAll('.unit-toggle button[data-unit="ev"]').forEach(btn => {
                    btn.click();
                });
            });
        </script>
        </body>
        '''
    )
    
    return html_content
