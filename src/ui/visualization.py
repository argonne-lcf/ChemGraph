"""3-D molecular structure visualisation components for the Streamlit UI."""

import json
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
from ase.data import chemical_symbols

from chemgraph.tools.ase_tools import create_ase_atoms, create_xyz_string

# ---------------------------------------------------------------------------
# Optional stmol / py3Dmol availability
# ---------------------------------------------------------------------------

try:
    import stmol

    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def warn_stmol_unavailable() -> None:
    """Display a one-time warning when stmol is not installed."""
    if not STMOL_AVAILABLE:
        st.warning("**stmol** not available -- falling back to text/table view.")
        st.info("To enable 3D visualization, install with: `pip install stmol`")


def create_ase_atoms_with_streamlit_error(atomic_numbers, positions):
    """Wrapper for ``create_ase_atoms`` that shows errors via Streamlit."""
    atoms = create_ase_atoms(atomic_numbers, positions)
    if atoms is None:
        st.error("Error creating ASE Atoms object")
    return atoms


def display_molecular_structure(atomic_numbers, positions, title="Structure") -> bool:
    """Render an interactive 3-D molecular viewer with info panel.

    Returns ``True`` on success, ``False`` on error.
    """
    try:
        atoms = create_ase_atoms_with_streamlit_error(atomic_numbers, positions)
        if atoms is None:
            return False

        xyz_string = create_xyz_string(atomic_numbers, positions)
        if xyz_string is None:
            return False

        st.subheader(f"\U0001f9ec {title}")
        col1, col2 = st.columns([2, 1])

        # 3-D panel --------------------------------------------------------
        with col1:
            if STMOL_AVAILABLE:
                style_options = ["ball_and_stick", "stick", "sphere", "wireframe"]
                selected_style = st.selectbox(
                    "Visualization Style",
                    style_options,
                    key=f"style_{uuid4().hex}",
                )

                try:
                    import py3Dmol

                    view = py3Dmol.view(width=500, height=400)
                    view.addModel(xyz_string, "xyz")

                    if selected_style == "ball_and_stick":
                        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
                    elif selected_style == "stick":
                        view.setStyle({"stick": {}})
                    elif selected_style == "sphere":
                        view.setStyle({"sphere": {}})
                    elif selected_style == "wireframe":
                        view.setStyle({"line": {}})
                    else:
                        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})

                    view.zoomTo()
                    stmol.showmol(view, height=400, width=500)

                except Exception as viz_error:
                    st.error(f"3D visualization error: {viz_error}")
                    st.info("Falling back to table view...")
                    _render_structure_table(atomic_numbers, positions)
            else:
                st.info("3-D viewer unavailable; showing raw XYZ and table.")
                with st.expander("\U0001f4c4 XYZ Format", expanded=True):
                    st.code(xyz_string, language="text")
                _render_structure_table(atomic_numbers, positions)

        # Info panel -------------------------------------------------------
        with col2:
            _render_structure_info(atoms, atomic_numbers, positions, xyz_string, title)

        return True
    except Exception as exc:
        st.error(f"Error displaying structure: {exc}")
    return False


def visualize_trajectory(traj):
    """Create an animated py3Dmol view from an ASE ``Trajectory``."""
    import py3Dmol

    xyz_frames = []
    for i, atoms in enumerate(traj):
        symbols = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        lines = [str(len(symbols)), f"Frame {i}"]
        lines += [
            f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, pos)
        ]
        xyz_frames.append("\n".join(lines))
    xyz_str = "\n".join(xyz_frames)

    view = py3Dmol.view(width=500, height=400)
    view.addModelsAsFrames(xyz_str, "xyz")

    view.setViewStyle({"style": "outline", "width": 0.05})
    view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    view.zoomTo()
    view.animate({"loop": "Forward", "interval": 100})

    return view


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_structure_table(atomic_numbers, positions) -> None:
    """Render a DataFrame table of atom positions."""
    data = []
    for idx, (num, pos) in enumerate(zip(atomic_numbers, positions), 1):
        sym = chemical_symbols[num] if num < len(chemical_symbols) else f"X{num}"
        data.append(
            {
                "Atom": idx,
                "Element": sym,
                "X": f"{pos[0]:.4f}",
                "Y": f"{pos[1]:.4f}",
                "Z": f"{pos[2]:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(data), height=350, use_container_width=True)


def _render_structure_info(atoms, atomic_numbers, positions, xyz_string, title) -> None:
    """Render the info/download panel beside the 3-D viewer."""
    st.markdown("**Structure Information**")
    st.write(f"- **Atoms:** {len(atoms)}")
    st.write(f"- **Formula:** {atoms.get_chemical_formula()}")

    # Composition
    composition: dict[str, int] = {}
    for atom in atoms:
        composition[atom.symbol] = composition.get(atom.symbol, 0) + 1
    st.write("**Composition:**")
    for elem, count in sorted(composition.items()):
        st.write(f"  \u2022 {elem}: {count}")

    # Total mass
    try:
        total_mass = atoms.get_masses().sum()
        st.write(f"**Total Mass:** {total_mass:.2f} amu")
    except Exception:
        st.write("**Total Mass:** Not available")

    # Center of mass
    try:
        com = atoms.get_center_of_mass()
        st.write("**Center of Mass:**")
        st.write(f"  [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f} ] \u00c5")
    except Exception:
        st.write("**Center of Mass:** Not available")

    # Additional properties
    with st.expander("\U0001f52c Additional Properties"):
        try:
            pos = atoms.positions
            com = atoms.get_center_of_mass()
            distances = np.linalg.norm(pos - com, axis=1)
            st.write(f"**Max distance from COM:** {distances.max():.3f} \u00c5")
            st.write(f"**Min distance from COM:** {distances.min():.3f} \u00c5")

            cell = atoms.get_cell()
            if np.any(cell.lengths()):
                st.write(f"**Cell lengths:** {cell.lengths()}")
                st.write(f"**Cell angles:** {cell.angles()}")
            else:
                st.write("**Cell:** non-periodic")
        except Exception as prop_error:
            st.write(f"Error calculating properties: {prop_error}")

    # Downloads
    st.write("**Download:**")
    st.download_button(
        "\U0001f4c4 XYZ File",
        xyz_string,
        f"{title.lower().replace(' ', '_')}.xyz",
        mime="chemical/x-xyz",
        key=f"xyz_download_{uuid4().hex}",
    )

    structure_json = json.dumps(
        {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "formula": atoms.get_chemical_formula(),
            "symbols": atoms.get_chemical_symbols(),
        },
        indent=2,
    )
    st.download_button(
        "\U0001f4cb JSON Data",
        structure_json,
        f"{title.lower().replace(' ', '_')}.json",
        mime="application/json",
        key=f"json_download_{uuid4().hex}",
    )
