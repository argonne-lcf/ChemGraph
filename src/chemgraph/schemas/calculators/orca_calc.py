# Keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/orca.html#ORCA
# Orca parameters for CompChemAgent

from typing import Optional
from pydantic import BaseModel, Field
from ase.calculators.orca import ORCA, OrcaProfile
import warnings
import os
import shutil


class OrcaCalc(BaseModel):
    """ORCA quantum chemistry calculator configuration.

    This class defines the configuration parameters for ORCA quantum chemistry
    calculations. It supports various quantum chemical methods and basis sets
    through the ORCA program.

    Parameters
    ----------
    calculator_type : str, optional
        Calculator type. Currently supports only 'orca', by default 'orca'
    charge : int, optional
        Total charge of the system, by default 0
    multiplicity : int, optional
        Total multiplicity of the system, by default 1
    orcasimpleinput : str, optional
        ORCA input keywords specifying method and basis set,
        by default 'B3LYP def2-TZVP'
    orcablocks : str, optional
        Additional ORCA block settings, by default '%pal nprocs 1 end'
    directory : str, optional
        Working directory for ORCA calculations, by default '.'
    profile : OrcaProfile, optional
        Optional ORCA profile configuration, by default None
    """

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
        """Get an ASE-compatible ORCA calculator instance.

        This method creates and returns an ORCA calculator instance with the
        specified configuration. It automatically searches for the ORCA executable
        if no profile is provided.

        Returns
        -------
        ORCA
            An ASE-compatible ORCA calculator instance

        Raises
        ------
        ValueError
            If an invalid calculator_type is specified
        """
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
