# pip install py3Dmol==2.0.0.post2
# ipyspeck(==0.6.1)
# ipywidgets(==7.6.3)
# ipython_genutils
# stmol

# ===== QM helpers
from ase.optimize import BFGS
from ase.vibrations import Vibrations
import numpy as np
import pandas as pd
#from xtb_ase import XTB
from xtb.ase.calculator import XTB
from ase.calculators.emt import EMT
from ase.vibrations.infrared import Infrared
from ase import Atoms
from ase.build import molecule
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
import py3Dmol
from stmol import showmol


def get_vibrations(atoms) -> Vibrations:
    """Calculate and return vibrations for a given molecule.
    
    Args:
        atoms: ASE Atoms object with calculator attached
        
    Returns:
        vib: ASE Vibrations object with calculated normal modes
    """
    # Optimize geometry first
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    
    # Calculate vibrations
    vib = Vibrations(atoms)
    vib.run()
    
    return vib


def visualize_trajectory(traj):
    """Create an animated 3D visualization of a trajectory.
    
    Args:
        traj: ASE Trajectory object
        
    Returns:
        view: py3Dmol view object with animated trajectory
    """
    # Convert all frames to a single multi-model XYZ string
    xyz_frames = []
    for i, atoms in enumerate(traj):
        symbols = atoms.get_chemical_symbols()
        pos = atoms.get_positions()  # Å
        lines = [str(len(symbols)), f'Frame {i}']
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, pos)]
        xyz_frames.append("\n".join(lines))
    xyz_str = "\n".join(xyz_frames)

    # Initialize viewer and add frames
    view = py3Dmol.view(width=800, height=400)
    view.addModelsAsFrames(xyz_str, 'xyz')   # load all frames at once

    # Style & camera
    view.setViewStyle({"style": "outline", "width": 0.05})
    view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    view.zoomTo()

    # Animate (interval in ms)
    view.animate({"loop": "Forward", "interval": 100})
    
    return view



# ===== Begin Streamlit
import streamlit as st

# Run initial calculation
h2o = molecule('H2O')
h2o.calc = EMT()

# Get vibrations and frequencies
vib = get_vibrations(h2o)
freq = vib.get_frequencies()
energies = vib.get_energies()
n_modes = len(freq)

# Display frequencies table
st.write("## Vibrational Modes")
freq_table = pd.DataFrame({
    'Mode': range(n_modes),
    'Frequency (cm⁻¹)': freq,
    'Energy (eV)': energies
})
st.table(freq_table)

# Mode selection dropdown
mode_options = [f"({i}) - {freq[i]:.1f} cm⁻¹ - {energies[i]:.3f} eV" 
                for i in range(n_modes)]
selected_mode = st.selectbox("Select vibrational mode to visualize:", mode_options)

# Get selected index and visualize
if selected_mode:
    mode_idx = int(selected_mode.split('(')[1].split(')')[0])
    vib.write_mode(mode_idx)
    traj = Trajectory(f'vib.{mode_idx}.traj')
    view = visualize_trajectory(traj)
    showmol(view, height = 400, width=800)

