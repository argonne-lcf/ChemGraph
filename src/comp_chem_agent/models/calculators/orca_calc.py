# Keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/orca.html#ORCA
# Orca parameters for CompChemAgent

from typing import Optional
from pydantic import BaseModel, Field
from ase.calculators.orca import ORCA


class OrcaCalc(BaseModel):
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
    profile: Optional[str] = Field(
        default=None, description="Optional ORCA profile configuration."
    )

    def get_calculator(self):
        """Returns an ASE-compatible ORCA calculator instance."""
        if self.calculator_type != "orca":
            raise ValueError(
                "Invalid calculator_type. The only valid option is 'orca'."
            )

        return ORCA(
            charge=self.charge,
            mult=self.multiplicity,
            orcasimpleinput=self.orcasimpleinput,
            orcablocks=self.orcablocks,
            directory=self.directory,
            profile=self.profile,
        )
