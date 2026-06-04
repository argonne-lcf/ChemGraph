"""Core simulation functions for phonopy calculations.

This module provides the core logic for running phonon calculations
using phonopy and ASE calculators.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

warnings.filterwarnings("ignore", module="phonopy")

from chemgraph.schemas.phonopy_schema import PhonopyInputSchema, PhonopyOutputSchema
from chemgraph.tools.ase_core import _resolve_path, load_calculator

def _auto_supercell_matrix(atoms, target_length: float = 10.0, is_2d: bool = False) -> List[int]:
    """Calculate the optimal supercell matrix to ensure cell dimensions >= target_length."""
    lengths = atoms.cell.lengths()
    matrix = [1, 1, 1]
    for i in range(3):
        if is_2d and i == 2:
            matrix[i] = 1
        else:
            if lengths[i] > 0:
                matrix[i] = int(np.ceil(target_length / lengths[i]))
            else:
                matrix[i] = 1
    return matrix


def run_phonopy_core(params: PhonopyInputSchema) -> dict:
    """Run a phonopy calculation using an ASE calculator for forces.

    Parameters
    ----------
    params : PhonopyInputSchema
        Fully validated phonopy input.

    Returns
    -------
    dict
        Result payload including file paths and status.
    """
    try:
        import phonopy
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        return {
            "status": "failure",
            "error_type": "ImportError",
            "message": "Phonopy is not installed. Please install it using 'pip install phonopy'."
        }

    from ase.io import read
    from ase import Atoms

    start_time = time.time()
    input_structure_file = params.input_structure_file
    
    if not os.path.isfile(input_structure_file):
        return {
            "status": "failure",
            "error_type": "FileNotFoundError",
            "message": f"Input structure file {input_structure_file} does not exist.",
        }

    try:
        atoms = read(input_structure_file)
    except Exception as e:
        return {
            "status": "failure",
            "error_type": type(e).__name__,
            "message": f"Cannot read {input_structure_file} using ASE. Exception from ASE: {e}",
        }

    # Determine Supercell Matrix
    if params.supercell_matrix:
        supercell_matrix_list = params.supercell_matrix
        supercell_matrix = np.diag(supercell_matrix_list)
    else:
        supercell_matrix_list = _auto_supercell_matrix(atoms, target_length=params.supercell_target_length, is_2d=params.is_2d)
        supercell_matrix = np.diag(supercell_matrix_list)

    # Initialize Phonopy
    unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                            cell=atoms.cell[:],
                            scaled_positions=atoms.get_scaled_positions())
    
    phonon = Phonopy(unitcell,
                     supercell_matrix=supercell_matrix,
                     primitive_matrix='P',
                     symprec=params.symprec)
                     
    phonon.generate_displacements(distance=0.01)
    supercells = phonon.supercells_with_displacements

    # Load ASE Calculator
    try:
        calculator_dict = params.calculator.model_dump()
        calc, system_info, calc_model = load_calculator(calculator_dict)
    except Exception as e:
        return {
            "status": "failure",
            "error_type": "ValueError",
            "message": f"Error loading calculator: {e}",
        }
        
    if calc is None:
        return {
            "status": "failure",
            "error_type": "ValueError",
            "message": "Unsupported calculator.",
        }

    # Calculate Forces for Displaced Supercells
    force_sets = []
    for scell in supercells:
        if scell is None:
            continue
        ase_scell = Atoms(symbols=scell.symbols,
                          scaled_positions=scell.scaled_positions,
                          cell=scell.cell,
                          pbc=True)
        ase_scell.info.update(system_info)
        ase_scell.calc = calc
        forces = ase_scell.get_forces()
        force_sets.append(forces)

    # Set forces and produce force constants
    phonon.forces = force_sets
    phonon.produce_force_constants()

    mol_stem = Path(input_structure_file).stem if input_structure_file else "structure"
    
    # Save phonopy yaml
    phonopy_yaml_path = _resolve_path(f"phonopy_{mol_stem}.yaml")
    phonon.save(phonopy_yaml_path)

    fc_file_path = None
    poscar_file_paths = []
    if params.save_vasp_files:
        from phonopy.file_IO import write_FORCE_CONSTANTS
        from phonopy.interface.vasp import write_vasp
        
        fc_file_path = _resolve_path(f"FORCE_CONSTANTS_{mol_stem}")
        write_FORCE_CONSTANTS(phonon.force_constants, filename=fc_file_path)
        
        sposcar_path = _resolve_path(f"SPOSCAR_{mol_stem}")
        write_vasp(sposcar_path, phonon.supercell)
        poscar_file_paths.append(sposcar_path)
        
        for i, cell in enumerate(phonon.supercells_with_displacements):
            if cell is not None:
                p_path = _resolve_path(f"POSCAR-{i+1:03d}_{mol_stem}")
                write_vasp(p_path, cell)
                poscar_file_paths.append(p_path)

    dos_plot_path = None
    tp_plot_path = None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Calculate DOS
    if params.calculate_dos:
        phonon.run_mesh(params.mesh, is_gamma_center=True)
        phonon.run_total_dos()
        dos_dict = phonon.get_total_dos_dict()
        
        fig, ax = plt.subplots()
        ax.plot(dos_dict['frequency_points'], dos_dict['total_dos'])
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Total DOS")
        ax.set_title("Phonon Density of States")
        ax.grid(True)
        dos_plot_path = _resolve_path(f"dos_{mol_stem}.png")
        fig.savefig(dos_plot_path, dpi=300)
        plt.close(fig)

    # Calculate Thermal Properties
    if params.calculate_thermal_properties:
        phonon.run_thermal_properties()
        tp_dict = phonon.get_thermal_properties_dict()
        
        fig, ax1 = plt.subplots()
        ax1.plot(tp_dict['temperatures'], tp_dict['free_energy'], 'r-', label='Free Energy (kJ/mol)')
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Free Energy (kJ/mol)", color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        
        ax2 = ax1.twinx()
        ax2.plot(tp_dict['temperatures'], tp_dict['entropy'], 'b--', label='Entropy (J/K/mol)')
        ax2.plot(tp_dict['temperatures'], tp_dict['heat_capacity'], 'g-.', label='Heat Capacity (J/K/mol)')
        ax2.set_ylabel("Entropy / Heat Capacity", color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        fig.tight_layout()
        tp_plot_path = _resolve_path(f"thermal_properties_{mol_stem}.png")
        fig.savefig(tp_plot_path, dpi=300)
        plt.close(fig)

    # Calculate Band Structure
    bs_plot_path = None
    band_yaml_path = None
    if params.calculate_band_structure:
        band_yaml_path = _resolve_path(f"band_{mol_stem}.yaml")
        if params.band_paths:
            from phonopy.phonon.band_structure import get_band_qpoints
            bands = get_band_qpoints(params.band_paths, npoints=params.band_npoints)
            phonon.run_band_structure(
                bands,
                labels=params.band_labels,
                with_eigenvectors=False
            )
            phonon.write_yaml_band_structure(filename=band_yaml_path)
        else:
            # Fallback to automatic path generation using seekpath
            phonon.auto_band_structure(npoints=params.band_npoints, with_eigenvectors=False, plot=False, write_yaml=True, filename=band_yaml_path)
            
        bs_plot_path = _resolve_path(f"band_structure_{mol_stem}.png")
        if params.dft_phonon_file and os.path.isfile(params.dft_phonon_file):
            # Custom comparison plot
            bs_dict = phonon.get_band_structure_dict()
            phonopy_distances = bs_dict['distances']  # list of 1D arrays
            phonopy_frequencies = bs_dict['frequencies']  # list of 2D arrays: (npoints, nbands)
            
            fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
            
            # Style the spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2c3e50')
            ax.spines['bottom'].set_color('#2c3e50')
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            
            color_phonopy = '#1abc9c'  # Deep teal/emerald for FAIRChem
            color_dft = '#e74c3c'      # Coral/red for DFT
            
            # Read DFT data
            try:
                with open(params.dft_phonon_file) as f:
                    dft_lines = f.readlines()
                
                dft_segments = []
                current_segment = []
                for line in dft_lines:
                    line_str = line.strip()
                    if not line_str:
                        if current_segment:
                            dft_segments.append(current_segment)
                            current_segment = []
                        continue
                    if line_str.startswith('#'):
                        continue
                    parts = line_str.split()
                    if len(parts) >= 2:
                        try:
                            current_segment.append([float(x) for x in parts])
                        except ValueError:
                            pass
                if current_segment:
                    dft_segments.append(current_segment)
                dft_segments = [seg for seg in dft_segments if len(seg) > 0]
                
                is_column_based = False
                if dft_segments and len(dft_segments[0][0]) > 2:
                    is_column_based = True
                
                if is_column_based:
                    # Column-based format: x y1 y2 y3 ...
                    data_pts = []
                    for seg in dft_segments:
                        data_pts.extend(seg)
                    data_pts = np.array(data_pts)
                    dft_x = data_pts[:, 0]
                    nbands_dft = data_pts.shape[1] - 1
                    for b_idx in range(nbands_dft):
                        label = 'DFT (reference)' if b_idx == 0 else ""
                        ax.plot(dft_x, data_pts[:, b_idx + 1], color=color_dft, linestyle='--', linewidth=1.6, label=label, alpha=0.8)
                else:
                    # Segment-based format: x y
                    npaths = len(phonopy_distances)
                    nbands_dft = len(dft_segments) // npaths
                    for b_idx in range(nbands_dft):
                        seg_x = []
                        seg_y = []
                        for p_idx in range(npaths):
                            if b_idx * npaths + p_idx < len(dft_segments):
                                seg = dft_segments[b_idx * npaths + p_idx]
                                seg_x.extend([pt[0] for pt in seg])
                                seg_y.extend([pt[1] for pt in seg])
                        label = 'DFT (reference)' if b_idx == 0 else ""
                        ax.plot(seg_x, seg_y, color=color_dft, linestyle='--', linewidth=1.6, label=label, alpha=0.8)
            except Exception as dft_err:
                print(f"Error parsing DFT phonon file: {dft_err}")
            
            # Plot Phonopy (FAIRChem/OMat)
            nbands_phonopy = phonopy_frequencies[0].shape[1]
            flat_distances = []
            for p_idx, dist_array in enumerate(phonopy_distances):
                for b_idx in range(nbands_phonopy):
                    label = 'FAIRChem (OMat)' if (p_idx == 0 and b_idx == 0) else ""
                    ax.plot(dist_array, phonopy_frequencies[p_idx][:, b_idx], color=color_phonopy, linestyle='-', linewidth=2.2, label=label, alpha=0.9)
                flat_distances.extend(dist_array)
            
            # Draw vertical lines for high-symmetry points
            high_sym_positions = [phonopy_distances[0][0]]
            for dist_array in phonopy_distances:
                high_sym_positions.append(dist_array[-1])
            
            for pos in high_sym_positions:
                ax.axvline(x=pos, color='#bdc3c7', linestyle=':', linewidth=1.2)
            
            # Set x-ticks and limits
            ax.set_xticks(high_sym_positions)
            labels = None
            if params.band_labels:
                labels = []
                for label in params.band_labels:
                    if label.lower() in ('gamma', 'g'):
                        labels.append(r'$\Gamma$')
                    else:
                        labels.append(label)
            elif hasattr(phonon, 'band_structure') and getattr(phonon.band_structure, 'labels', None) is not None:
                labels = phonon.band_structure.labels
                
            if labels and len(labels) == len(high_sym_positions):
                ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_xlim(0, max(flat_distances))
            
            # Labels and titles
            ax.set_ylabel('Frequency (THz)', fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(f'Phonon Dispersion: {mol_stem}', fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
            ax.text(0.5, 1.02, f'FAIRChem (OMat) vs DFT (from {os.path.basename(params.dft_phonon_file)})', transform=ax.transAxes, 
                    ha='center', fontsize=11, color='#7f8c8d')
            
            # Grid and Legend
            ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#ecf0f1')
            ax.legend(frameon=True, facecolor='white', edgecolor='#ecf0f1', fontsize=11, loc='upper right', shadow=True)
            
            plt.tight_layout()
            fig.savefig(bs_plot_path, dpi=300)
            plt.close(fig)
        else:
            bs_plot = phonon.plot_band_structure()
            bs_plot.savefig(bs_plot_path, dpi=300)
            bs_plot.close()

    end_time = time.time()
    wall_time = end_time - start_time

    # Save output schema
    output_results_file = _resolve_path(params.output_results_file)
    
    simulation_output = PhonopyOutputSchema(
        input_structure_file=input_structure_file,
        simulation_input=params,
        success=True,
        supercell_matrix_used=supercell_matrix_list,
        thermal_properties_plot=tp_plot_path,
        dos_plot=dos_plot_path,
        band_structure_plot=bs_plot_path,
        phonopy_yaml=phonopy_yaml_path,
        wall_time=wall_time,
    )
    
    with open(output_results_file, "w", encoding="utf-8") as wf:
        wf.write(simulation_output.model_dump_json(indent=4))

    return {
        "status": "success",
        "message": f"Phonopy simulation completed. Results saved to {os.path.abspath(output_results_file)}",
        "phonopy_yaml": os.path.abspath(phonopy_yaml_path),
        "dos_plot": os.path.abspath(dos_plot_path) if dos_plot_path else None,
        "thermal_properties_plot": os.path.abspath(tp_plot_path) if tp_plot_path else None,
        "band_structure_plot": os.path.abspath(bs_plot_path) if bs_plot_path else None,
        "band_yaml": os.path.abspath(band_yaml_path) if band_yaml_path else None,
        "force_constants_file": os.path.abspath(fc_file_path) if fc_file_path else None,
        "poscar_files": [os.path.abspath(p) for p in poscar_file_paths] if poscar_file_paths else None,
        "supercell_matrix_used": supercell_matrix_list,
    }
