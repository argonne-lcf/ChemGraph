"""Input schema for the molecular docking tool."""


from pydantic import BaseModel, Field


class docking_input_schema(BaseModel):
    """Parameters for docking a small-molecule candidate into a receptor."""

    candidate: str = Field(
        description=(
            "Molecule to dock, given as a SMILES string, a molecule name "
            "(e.g. 'aspirin'), or a PubChem CID (e.g. '2244')."
        )
    )
    receptor: str = Field(
        default="vancomycin",
        description=(
            "Docking target: 'vancomycin' (bundled default) or a path to a "
            "prepared rigid receptor .pdbqt file."
        ),
    )
    n_poses: int = Field(
        default=10,
        description="Number of docked poses to generate.",
    )
    center: list[float] | None = Field(
        default=None,
        description=(
            "Search-box center [x, y, z] in Angstrom. Required for a custom "
            "receptor; ignored for the bundled 'vancomycin' target."
        ),
    )
    box_size: list[float] | None = Field(
        default=None,
        description=(
            "Search-box size [x, y, z] in Angstrom. Required for a custom "
            "receptor; ignored for the bundled 'vancomycin' target."
        ),
    )
    exhaustiveness: int = Field(
        default=8,
        description="AutoDock Vina search exhaustiveness (higher = more thorough).",
    )
