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
    mol_stem = Path(input_structure_file).stem if input_structure_file else "structure"
    
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

    orig_lengths = list(atoms.cell.lengths())
    orig_angles = list(atoms.cell.angles())
    new_lengths = None
    new_angles = None

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

    # Auto-detect if the structure is 2D
    is_2d_detected = False
    lengths = atoms.cell.lengths()
    if len(lengths) == 3:
        # Check Z axis (axis 2) for vacuum gap
        scaled_pos = atoms.get_scaled_positions()
        coords = np.sort(scaled_pos[:, 2])
        gaps = np.diff(coords)
        wrap_gap = 1.0 - (coords[-1] - coords[0]) if len(coords) > 0 else 1.0
        max_fractional_gap = max(gaps.max() if len(gaps) > 0 else 0, wrap_gap)
        max_physical_gap = max_fractional_gap * lengths[2]
        if max_physical_gap >= 6.0:
            is_2d_detected = True

    is_2d_effective = params.is_2d or is_2d_detected

    # Minimize structure if requested
    minimization_info_str = "Disabled"
    minimized_structure_file = None
    minimization_log_file = None
    if params.minimize_structure:
        from ase.optimize import BFGS
        from ase.io import write as ase_write
        try:
            atoms.calc = calc
            opt_log_path = _resolve_path(f"minimization_{mol_stem}.log")
            
            mode = params.relaxation_mode.lower()
            if mode == "full":
                from ase.filters import UnitCellFilter
                if is_2d_effective:
                    mask = [True, True, False, False, False, True]
                    target = UnitCellFilter(atoms, mask=mask)
                    minimization_info_str = "Enabled (Full relaxation of both atomic positions and lattice parameters, with Z-axis fixed for 2D structure)"
                else:
                    target = UnitCellFilter(atoms)
                    minimization_info_str = "Enabled (Full relaxation of both atomic positions and lattice parameters)"
            elif mode == "cell":
                from ase.filters import UnitCellFilter
                from ase.constraints import FixAtoms
                atoms.set_constraint(FixAtoms(mask=[True] * len(atoms)))
                if is_2d_effective:
                    mask = [True, True, False, False, False, True]
                    target = UnitCellFilter(atoms, mask=mask)
                    minimization_info_str = "Enabled (Relaxation of only lattice parameters, keeping atomic positions fixed, with Z-axis fixed for 2D structure)"
                else:
                    target = UnitCellFilter(atoms)
                    minimization_info_str = "Enabled (Relaxation of only lattice parameters, keeping atomic positions fixed)"
            elif mode == "atoms":
                target = atoms
                minimization_info_str = "Enabled (Relaxation of only atomic positions, keeping lattice parameters fixed)"
            else:
                target = atoms
                minimization_info_str = "Enabled (Relaxation of only atomic positions, keeping lattice parameters fixed)"

            dyn = BFGS(target, logfile=opt_log_path)
            converged = dyn.run(fmax=2e-4, steps=500)
            
            # Remove constraint if it was set
            if hasattr(atoms, "set_constraint"):
                atoms.set_constraint()
                
            atoms.calc = None
            new_lengths = list(atoms.cell.lengths())
            new_angles = list(atoms.cell.angles())
            
            # Save the minimized structure
            minimized_path = _resolve_path(f"POSCAR_minimized_{mol_stem}")
            ase_write(minimized_path, atoms, format="vasp")
            minimized_structure_file = minimized_path
            minimization_log_file = opt_log_path
            
            if converged:
                minimization_info_str += f" (converged to fmax = 2e-4 eV/Å within 500 steps, minimized structure saved to {os.path.basename(minimized_path)})"
            else:
                minimization_info_str += f" (did NOT converge to fmax = 2e-4 eV/Å within 500 steps, using non-minimized structure)"
        except Exception as opt_err:
            if hasattr(atoms, "set_constraint"):
                atoms.set_constraint()
            if atoms.calc is not None:
                atoms.calc = None
            print(f"Structure minimization failed: {opt_err}. Continuing with original structure.")
            minimization_info_str = f"Failed (raised error: {opt_err}, using original structure)"

    # Determine Supercell Matrix
    if params.supercell_matrix:
        supercell_matrix_list = params.supercell_matrix
        supercell_matrix = np.diag(supercell_matrix_list)
    else:
        supercell_matrix_list = _auto_supercell_matrix(atoms, target_length=params.supercell_target_length, is_2d=is_2d_effective)
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
        
        poscar_dir = _resolve_path("poscar_displacements")
        os.makedirs(poscar_dir, exist_ok=True)
        for i, cell in enumerate(phonon.supercells_with_displacements):
            if cell is not None:
                p_path = os.path.join(poscar_dir, f"POSCAR-{i+1:03d}_{mol_stem}")
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
    band_dat_path = None
    info_file_path = None
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
            if is_2d_effective:
                from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
                try:
                    bands, labels, path_connections = get_band_qpoints_by_seekpath(
                        phonon.primitive, npoints=params.band_npoints
                    )
                    
                    # Reconstruct segments and labels
                    seg_labels = []
                    lbl_idx = 0
                    for i in range(len(bands)):
                        start_lbl = labels[lbl_idx]
                        end_lbl = labels[lbl_idx + 1]
                        seg_labels.append((start_lbl, end_lbl))
                        if i < len(path_connections) and path_connections[i]:
                            lbl_idx += 1
                        else:
                            lbl_idx += 2
                    
                    bands_filtered = []
                    labels_filtered = []
                    kept_indices = set()
                    
                    for i in range(len(bands)):
                        start_q = bands[i][0]
                        end_q = bands[i][-1]
                        if np.isclose(start_q[2], 0, atol=1e-4) and np.isclose(end_q[2], 0, atol=1e-4):
                            bands_filtered.append(bands[i])
                            start_lbl, end_lbl = seg_labels[i]
                            if not labels_filtered:
                                labels_filtered.append(start_lbl)
                            else:
                                if i > 0 and path_connections[i-1] and (i-1) in kept_indices:
                                    pass
                                else:
                                    labels_filtered.append(start_lbl)
                            labels_filtered.append(end_lbl)
                            kept_indices.add(i)
                    
                    if bands_filtered:
                        phonon.run_band_structure(
                            bands_filtered,
                            labels=labels_filtered,
                            with_eigenvectors=False
                        )
                        phonon.write_yaml_band_structure(filename=band_yaml_path)
                    else:
                        phonon.auto_band_structure(npoints=params.band_npoints, with_eigenvectors=False, plot=False, write_yaml=True, filename=band_yaml_path)
                except Exception as seek_err:
                    print(f"Error filtering 2D band paths: {seek_err}. Falling back to default seekpath.")
                    phonon.auto_band_structure(npoints=params.band_npoints, with_eigenvectors=False, plot=False, write_yaml=True, filename=band_yaml_path)
            else:
                phonon.auto_band_structure(npoints=params.band_npoints, with_eigenvectors=False, plot=False, write_yaml=True, filename=band_yaml_path)
            
        # Generate band .dat file using phonopy-bandplot --gnuplot command or Python fallback
        band_dat_path = _resolve_path(f"band_{mol_stem}.dat")
        try:
            import subprocess
            import sys
            bin_dir = os.path.dirname(sys.executable)
            bandplot_exe = os.path.join(bin_dir, "phonopy-bandplot")
            if not os.path.isfile(bandplot_exe):
                bandplot_exe = "phonopy-bandplot"
            with open(band_dat_path, "w") as f_dat:
                subprocess.run([bandplot_exe, "--gnuplot", band_yaml_path], stdout=f_dat, check=True)
            print(f"Generated band dat file via phonopy-bandplot: {band_dat_path}")
        except Exception as e:
            print(f"Failed to generate band dat file using subprocess: {e}. Falling back to Python implementation.")
            try:
                bs_dict = phonon.get_band_structure_dict()
                phonopy_distances = bs_dict['distances']
                phonopy_frequencies = bs_dict['frequencies']
                nbands = phonopy_frequencies[0].shape[1]
                npaths = len(phonopy_distances)
                high_sym_positions = [phonopy_distances[0][0]] + [dist[-1] for dist in phonopy_distances]
                with open(band_dat_path, "w") as f_dat:
                    f_dat.write("# End points of segments:\n")
                    f_dat.write("#   " + "".join(f"{pos:10.8f} " for pos in high_sym_positions) + "\n")
                    for b_idx in range(nbands):
                        for p_idx in range(npaths):
                            dist_array = phonopy_distances[p_idx]
                            freq_array = phonopy_frequencies[p_idx][:, b_idx]
                            for d, f_val in zip(dist_array, freq_array):
                                f_dat.write(f"{d:f} {f_val:f}\n")
                            f_dat.write("\n")
                        f_dat.write("\n")
                print(f"Generated band dat file via fallback Python implementation: {band_dat_path}")
            except Exception as py_err:
                print(f"Failed to generate band dat file via fallback Python code: {py_err}")
                band_dat_path = None
            
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

    # Generate Phonon_Calculation_Info.md in English
    info_file_path = _resolve_path("Phonon_Calculation_Info.md")
    try:
        with open(info_file_path, "w", encoding="utf-8") as f_info:
            f_info.write("# Phonon Calculation Information\n\n")
            f_info.write("This file summarizes the input configurations and parameters used for this phonon calculation.\n\n")
            
            f_info.write("## 1. Calculator & Model Configuration\n\n")
            f_info.write(f"- **Calculator Type**: `{params.calculator.calculator_type}`\n")
            if hasattr(params.calculator, "model_name"):
                f_info.write(f"- **Model Name**: `{params.calculator.model_name}`\n")
            if hasattr(params.calculator, "task_name") and params.calculator.task_name:
                f_info.write(f"- **Task Name**: `{params.calculator.task_name}`\n")
            if hasattr(params.calculator, "device"):
                f_info.write(f"- **Device**: `{params.calculator.device}`\n")
            f_info.write(f"- **Structure Minimization**: `{minimization_info_str}`\n")
            if params.minimize_structure and new_lengths is not None:
                orig_len_str = ", ".join(f"{x:.3f}" for x in orig_lengths)
                new_len_str = ", ".join(f"{x:.3f}" for x in new_lengths)
                orig_ang_str = ", ".join(f"{x:.3f}" for x in orig_angles)
                new_ang_str = ", ".join(f"{x:.3f}" for x in new_angles)
                f_info.write(f"  - **Lattice Parameters (Before)**: a, b, c = [{orig_len_str}] Å; alpha, beta, gamma = [{orig_ang_str}]°\n")
                f_info.write(f"  - **Lattice Parameters (After)**: a, b, c = [{new_len_str}] Å; alpha, beta, gamma = [{new_ang_str}]°\n")
            f_info.write("\n")
            
            f_info.write("## 2. Supercell Configuration\n\n")
            f_info.write(f"- **Supercell Matrix**: `{supercell_matrix_list}`\n")
            if is_2d_effective:
                detection_source = "User specified" if params.is_2d else "Auto-detected"
                f_info.write(f"- **Dimensionality**: 2D (perpendicular cell dimension set to 1) [{detection_source}]\n")
            else:
                f_info.write("- **Dimensionality**: 3D bulk\n")
            f_info.write("\n")
            
            f_info.write("## 3. Reciprocal Space (k-space) High Symmetry Path\n\n")
            
            def clean_latex_label(lbl: str) -> str:
                lbl = lbl.replace("$", "")
                lbl = lbl.replace(r"\Gamma", "Gamma").replace("\\Gamma", "Gamma")
                lbl = lbl.replace(r"\Sigma", "Sigma").replace("\\Sigma", "Sigma")
                lbl = lbl.replace(r"\Delta", "Delta").replace("\\Delta", "Delta")
                lbl = lbl.replace(r"\Lambda", "Lambda").replace("\\Lambda", "Lambda")
                lbl = lbl.replace(r"\mathrm{", "").replace("\\mathrm{", "")
                lbl = lbl.replace("}", "")
                return lbl.strip()

            k_points_list = []
            if params.band_labels and params.band_paths:
                flat_qpoints = []
                for segment in params.band_paths:
                    for qpt in segment:
                        if not flat_qpoints or not np.allclose(flat_qpoints[-1], qpt):
                            flat_qpoints.append(qpt)
                if len(params.band_labels) == len(flat_qpoints):
                    for label, qpt in zip(params.band_labels, flat_qpoints):
                        k_points_list.append((label, list(qpt)))
                else:
                    for label in params.band_labels:
                        k_points_list.append((label, None))
            else:
                # Read from band_yaml_path if auto-generated via seekpath
                try:
                    import yaml
                    if band_yaml_path and os.path.isfile(band_yaml_path):
                        with open(band_yaml_path, "r", encoding="utf-8") as f_yaml:
                            yaml_data = yaml.safe_load(f_yaml)
                        labels_list = yaml_data.get("labels")
                        segment_nqpoint = yaml_data.get("segment_nqpoint")
                        phonon_data = yaml_data.get("phonon")
                        if labels_list and segment_nqpoint and phonon_data:
                            q_idx = 0
                            for seg_idx, (seg_labels, nq) in enumerate(zip(labels_list, segment_nqpoint)):
                                start_label = clean_latex_label(seg_labels[0])
                                end_label = clean_latex_label(seg_labels[1])
                                start_q = phonon_data[q_idx].get("q-position")
                                end_q = phonon_data[q_idx + nq - 1].get("q-position")
                                if not k_points_list or not np.allclose(k_points_list[-1][1], start_q):
                                    k_points_list.append((start_label, start_q))
                                k_points_list.append((end_label, end_q))
                                q_idx += nq
                except Exception as yaml_err:
                    print(f"Error parsing band.yaml for k-space info: {yaml_err}")

            if k_points_list:
                f_info.write("| Label | Coordinates [q_x, q_y, q_z] |\n")
                f_info.write("| :---: | :---: |\n")
                for label, qpt in k_points_list:
                    qpt_str = ", ".join(f"{x:.6f}" for x in qpt) if qpt is not None else "N/A"
                    f_info.write(f"| {label} | `[{qpt_str}]` |\n")
            else:
                f_info.write("- **k-space path**: Auto-generated via Seekpath (standard high symmetry path)\n")
            f_info.write("\n")
            
            f_info.write("## 4. Generated Output Files\n\n")
            f_info.write(f"- **Results summary (JSON)**: `{os.path.basename(params.output_results_file)}`\n")
            if band_yaml_path:
                f_info.write(f"- **Band YAML**: `{os.path.basename(band_yaml_path)}`\n")
            if band_dat_path:
                f_info.write(f"- **Band Dat**: `{os.path.basename(band_dat_path)}`\n")
            if bs_plot_path:
                f_info.write(f"- **Band Plot**: `{os.path.basename(bs_plot_path)}`\n")
            if dos_plot_path:
                f_info.write(f"- **DOS Plot**: `{os.path.basename(dos_plot_path)}`\n")
            if tp_plot_path:
                f_info.write(f"- **Thermal Properties Plot**: `{os.path.basename(tp_plot_path)}`\n")
            if minimized_structure_file:
                f_info.write(f"- **Minimized Structure (VASP)**: `{os.path.basename(minimized_structure_file)}`\n")
            if minimization_log_file:
                f_info.write(f"- **Minimization Log**: `{os.path.basename(minimization_log_file)}`\n")
    except Exception as info_err:
        print(f"Failed to generate Phonon_Calculation_Info.md: {info_err}")
        info_file_path = None

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
        band_yaml=band_yaml_path,
        band_dat=band_dat_path,
        calculation_info_file=info_file_path,
        phonopy_yaml=phonopy_yaml_path,
        force_constants_file=fc_file_path,
        poscar_files=poscar_file_paths if poscar_file_paths else None,
        minimized_structure_file=minimized_structure_file,
        minimization_log_file=minimization_log_file,
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
        "band_dat": os.path.abspath(band_dat_path) if band_dat_path else None,
        "calculation_info_file": os.path.abspath(info_file_path) if info_file_path else None,
        "force_constants_file": os.path.abspath(fc_file_path) if fc_file_path else None,
        "poscar_files": [os.path.abspath(p) for p in poscar_file_paths] if poscar_file_paths else None,
        "minimized_structure_file": os.path.abspath(minimized_structure_file) if minimized_structure_file else None,
        "minimization_log_file": os.path.abspath(minimization_log_file) if minimization_log_file else None,
        "supercell_matrix_used": supercell_matrix_list,
    }
