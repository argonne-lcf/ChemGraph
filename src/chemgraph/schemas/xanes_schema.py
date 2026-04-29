import os
from typing import Optional, Union

from pydantic import BaseModel, Field


class xanes_input_schema(BaseModel):
    """Input schema for a single XANES/FDMNES calculation."""

    input_structure_file: str = Field(
        description="Path to the input structure file (CIF, POSCAR, XYZ, etc.)."
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory to write FDMNES input files and results. "
            "Defaults to a subdirectory next to the input structure."
        ),
    )
    z_absorber: Optional[int] = Field(
        default=None,
        description=(
            "Atomic number of the X-ray absorbing atom. "
            "Defaults to the heaviest element in the structure."
        ),
    )
    radius: float = Field(
        default=6.0,
        description="Cluster radius in Angstrom for the FDMNES calculation.",
    )
    magnetism: bool = Field(
        default=False,
        description="Enable magnetic contributions in the FDMNES calculation.",
    )


class xanes_input_schema_ensemble(BaseModel):
    """Input schema for ensemble XANES/FDMNES calculations via Parsl."""

    input_structures: Union[str, list[str]] = Field(
        description=(
            "Path to a directory of structure files OR a list of individual file paths."
        ),
    )
    z_absorber: Optional[int] = Field(
        default=None,
        description=(
            "Atomic number of the X-ray absorbing atom. "
            "Defaults to the heaviest element in each structure."
        ),
    )
    radius: float = Field(
        default=6.0,
        description="Cluster radius in Angstrom for the FDMNES calculation.",
    )
    magnetism: bool = Field(
        default=False,
        description="Enable magnetic contributions in the FDMNES calculation.",
    )
    fdmnes_exe: str = Field(
        default_factory=lambda: os.environ.get("FDMNES_EXE", "fdmnes"),
        description=(
            "Path to the FDMNES executable. "
            "Defaults to the FDMNES_EXE environment variable, or 'fdmnes'."
        ),
    )


class mp_query_schema(BaseModel):
    """Input schema for fetching structures from Materials Project."""

    chemsys: list[str] = Field(
        description="Chemical formulas to search (e.g. ['Fe2O3', 'CoO']).",
    )
    mp_api_key: Optional[str] = Field(
        default=None,
        description=(
            "Materials Project API key. "
            "If not provided, falls back to the MP_API_KEY environment variable."
        ),
    )
    energy_above_hull: float = Field(
        default=0.001,
        description=(
            "Maximum energy above hull in eV/atom for filtering stable structures."
        ),
    )
