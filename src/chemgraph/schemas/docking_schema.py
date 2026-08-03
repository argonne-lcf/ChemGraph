"""Input schema for the molecular docking tool."""

from typing import Literal

from pydantic import BaseModel, Field


class docking_input_schema(BaseModel):
    """Parameters for docking a small-molecule candidate into a receptor."""

    candidate: str = Field(
        description=(
            "Molecule to dock, as a SMILES string, a molecule name "
            "(e.g. 'aspirin'), or a PubChem CID (e.g. '2244')."
        )
    )
    receptor: str = Field(
        description=(
            "Docking target: a path to a prepared rigid receptor '.pdbqt' file, "
            "or a SMILES/name/PubChem CID to build a small-molecule receptor."
        )
    )
    n_poses: int = Field(
        default=10,
        description="Number of docked poses to generate.",
    )
    exhaustiveness: int = Field(
        default=8,
        description="AutoDock Vina search exhaustiveness (higher = more thorough).",
    )
    site_detection: Literal["auto", "reference", "fpocket", "blind"] = Field(
        default="auto",
        description=(
            "How to place the search box when 'center'/'box_size' are not given: "
            "'reference' centers on a bound ligand (needs 'reference_ligand'); "
            "'fpocket' detects a pocket (needs fpocket installed); "
            "'blind' searches the whole receptor; "
            "'auto' tries reference -> fpocket -> blind."
        ),
    )
    center: list[float] | None = Field(
        default=None,
        description="Optional search-box center [x, y, z] in Angstrom (overrides site_detection).",
    )
    box_size: list[float] | None = Field(
        default=None,
        description="Optional search-box size [x, y, z] in Angstrom (overrides site_detection).",
    )
    reference_ligand: str | None = Field(
        default=None,
        description=(
            "Path to a bound-ligand structure to center the box on "
            "(used when site_detection is 'reference' or 'auto')."
        ),
    )
