"""A deliberately small set of workspace-scoped chemistry tools for the web UI."""

import os
from pathlib import Path


STRUCTURE_FORMATS = {
    ".xyz": "extxyz",
    ".pdb": "proteindatabank",
    ".cif": "cif",
    ".traj": "traj",
}


def workspace_path(root: Path, value: str, *, reading=False) -> str:
    """Reuse chemistry path resolution, then enforce the session boundary."""
    from chemgraph.tools.ase_core import _resolve_existing_path, _resolve_path

    resolve = _resolve_existing_path if reading else _resolve_path
    path = Path(resolve(value)).resolve()
    try:
        path.relative_to(root.resolve())
        if not reading:
            path.relative_to(Path(os.environ["CHEMGRAPH_LOG_DIR"]).resolve())
    except ValueError:
        raise ValueError(
            "Read files from this conversation and write new files in the current run directory."
        ) from None
    if reading and not path.is_file():
        raise ValueError("The requested workspace file does not exist.")
    return str(path)


def web_tools(
    workspace: Path, calculators: tuple[str, ...], *, guard=None, publish=None
):
    """Wrap approved tools without exposing shell, Python, or model-file loading."""
    from langchain_core.tools import tool
    from chemgraph.schemas.ase_input import ASEInputSchema
    from chemgraph.tools import ase_core, cheminformatics_core
    from chemgraph.tools.cheminformatics_tools import molecule_name_to_smiles
    from chemgraph.tools.generic_tools import calculator

    guard = guard or (lambda: None)
    publish = publish or (lambda _path: None)

    @tool
    def smiles_to_coordinate_file(
        smiles: str, output_file: str = "molecule.xyz"
    ) -> str:
        """Create an XYZ structure from SMILES inside the conversation workspace."""
        guard()
        output_file = workspace_path(workspace, output_file)
        if Path(output_file).suffix.lower() != ".xyz":
            raise ValueError("Coordinate output must be an XYZ file.")
        result = cheminformatics_core.smiles_to_coordinate_file_core(
            smiles, output_file=output_file
        )
        guard()
        publish(output_file)
        return result

    @tool
    def run_ase(ase_input: ASEInputSchema) -> dict:
        """Run an ASE calculation using an approved calculator and workspace files."""
        guard()
        data = ase_input.model_copy(deep=True)
        calc = data.calculator
        if calc.calculator_type not in calculators:
            raise ValueError(
                f"Choose an approved calculator: {', '.join(calculators)}."
            )
        # MACE accepts arbitrary model paths/URLs; web runs use only the installed
        # foundation-model default. No user-specified serialized model is loaded.
        if getattr(calc, "model", None) is not None:
            raise ValueError(
                "Web calculations use the administrator's default foundation model."
            )
        data.input_structure_file = workspace_path(
            workspace, data.input_structure_file, reading=True
        )
        if Path(data.input_structure_file).suffix.lower() not in STRUCTURE_FORMATS:
            raise ValueError("Use an XYZ, PDB, CIF, or TRAJ structure.")
        data.output_results_file = workspace_path(workspace, data.output_results_file)
        if Path(data.output_results_file).suffix.lower() != ".json":
            raise ValueError("Simulation results must use a JSON filename.")
        result = ase_core.run_ase_core(data)
        guard()
        # Register only outputs owned by this chemistry operation. Internal
        # graph diagnostics and arbitrary JSON files are never auto-published.
        output = Path(data.output_results_file)
        stem = Path(data.input_structure_file).stem
        turn = Path(os.environ["CHEMGRAPH_LOG_DIR"])
        paths = [output, output.with_name(f"{stem}_opt.traj")]
        paths.extend(
            turn / name
            for name in (
                f"frequencies_{stem}.csv",
                f"ir_spectrum_{stem}.png",
                f"ir_spectrum_{stem}.csv",
                f"ir_peaks_{stem}.csv",
            )
        )
        paths.extend(turn.glob(f"{stem}_vib.*.traj"))
        for path in paths:
            if path.is_file():
                publish(path)
        return result

    @tool
    def read_workspace_file(filename: str) -> str:
        """Read an attached XYZ/PDB/CIF/TRAJ structure or JSON/CSV/TXT data file; large content is truncated."""
        guard()
        path = Path(workspace_path(workspace, filename, reading=True))
        if path.suffix.lower() == ".traj":
            return structure_xyz(path)[:50000]
        if path.suffix.lower() not in {".xyz", ".pdb", ".cif", ".json", ".csv", ".txt"}:
            raise ValueError("Choose a supported structure or text attachment.")
        with path.open(encoding="utf-8") as source:
            content = source.read(50001)
        return content[:50000] + (
            "\n[Content truncated]" if len(content) > 50000 else ""
        )

    @tool
    def extract_output_json(json_file: str) -> dict:
        """Read simulation results from a JSON file in this conversation."""
        path = workspace_path(workspace, json_file, reading=True)
        if Path(path).suffix.lower() != ".json":
            raise ValueError("Only JSON results can be read by this tool.")
        return ase_core.extract_output_json_core(path)

    return [
        molecule_name_to_smiles,
        smiles_to_coordinate_file,
        run_ase,
        extract_output_json,
        calculator,
        read_workspace_file,
    ]


def structure_xyz(path: Path) -> str:
    """Convert supported structures/trajectories to browser-readable XYZ frames."""
    import io
    from ase.io import iread, write

    fmt = STRUCTURE_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError("This artifact is not a supported molecular structure.")
    output = io.StringIO()
    for index, atoms in enumerate(iread(str(path), format=fmt, index=":")):
        if index >= 500 or len(atoms) > 20000:
            raise ValueError(
                "Structure preview exceeds 500 frames or 20,000 atoms; download the artifact instead."
            )
        write(output, atoms, format="xyz")
        if output.tell() > 25 * 1024 * 1024:
            raise ValueError(
                "Structure preview is too large; download the artifact instead."
            )
    return output.getvalue()
