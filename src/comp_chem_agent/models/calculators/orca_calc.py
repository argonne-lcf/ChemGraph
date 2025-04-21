# Keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/orca.html#ORCA
# Orca parameters for CompChemAgent

from typing import Optional
from pydantic import BaseModel, Field
from ase.calculators.orca import ORCA, OrcaProfile
import warnings
import os
import shutil


class OrcaCalc(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    calculator_type: str = Field(
        default="orca", description="Calculator type. Currently supports only 'orca'."
    )
    charge: int = Field(default=0, description="Total charge of the system.")
    multiplicity: int = Field(
        default=1, description="Total multiplicity of the system."
    )
    orcasimpleinput: str = Field(
        default="B3LYP def2-TZVP",
        description="ORCA input keywords specifying method and basis set.",
    )
    orcablocks: str = Field(
        default="%pal nprocs 1 end", description="Additional ORCA block settings."
    )
    directory: str = Field(
        default=".", description="Working directory for ORCA calculations."
    )
    profile: Optional[OrcaProfile] = Field(
        default=None, description="Optional ORCA profile configuration."
    )

    def get_calculator(self):
        """Returns an ASE-compatible ORCA calculator instance."""
        if self.calculator_type != "orca":
            raise ValueError(
                "Invalid calculator_type. The only valid option is 'orca'."
            )

        # Check if profile is provided, otherwise try to find orca executable
        if self.profile is None:
            # First check if orca is in PATH
            orca_path = shutil.which("orca")

            # If not in PATH, check common installation directories
            if not orca_path:
                common_paths = [
                    "/opt/orca",
                    "/usr/local/orca",
                    os.path.expanduser("~/orca"),
                ]

                for path in common_paths:
                    potential_path = os.path.join(path, "orca")
                    if os.path.isfile(potential_path) and os.access(
                        potential_path, os.X_OK
                    ):
                        orca_path = potential_path
                        break

            if orca_path:
                profile = OrcaProfile(command=orca_path)
                print(f"Found ORCA executable at: {orca_path}")
            else:
                warnings.warn(
                    "ORCA executable not found in PATH or common paths. Please provide the path "
                    "using profile=OrcaProfile(command='/path/to/orca')"
                )
                profile = None
        else:
            profile = self.profile

        return ORCA(
            charge=self.charge,
            mult=self.multiplicity,
            orcasimpleinput=self.orcasimpleinput,
            orcablocks=self.orcablocks,
            directory=self.directory,
            profile=profile,
        )
