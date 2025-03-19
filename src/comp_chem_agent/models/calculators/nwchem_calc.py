# Main keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/nwchem.html
# Parameters for NWChem calculator in CompChemAgent

from typing import Optional, Union, Dict
from pydantic import BaseModel, Field
from ase.calculators.nwchem import NWChem


class NWChemCalc(BaseModel):
    calculator_type: str = Field(
        default="nwchem", description="Calculator type. Currently supports only 'nwchem'."
    )
    theory: Optional[str] = Field(
        default="dft",
        description="NWChem module to be used. Options: 'dft', 'scf', 'mp2', 'ccsd', 'tce', 'tddft', 'pspw', 'band', 'paw'.",
    )
    xc: Optional[str] = Field(
        default="PBE",
        description="Exchange-correlation functional (only applicable for DFT calculations).",
    )
    basis: Optional[Union[str, Dict[str, str]]] = Field(
        default="6-31G",
        description="Basis set to use. Can be a string for all elements or a dictionary mapping elements to basis sets.",
    )
    kpts: Optional[Union[tuple, Dict[str, Union[int, str]]]] = Field(
        default=None, description="K-point mesh for periodic calculations."
    )
    directory: str = Field(default=".", description="Working directory for NWChem calculations.")
    mult: int = Field(default=1, description="Multiplicity")
    odft: str = Field(default=None, description="Set the calculation to be open-shell.")
    command: Optional[str] = Field(
        default=None,
        description="Command to execute NWChem (e.g., 'nwchem PREFIX.nwi > PREFIX.nwo').",
    )

    def get_calculator(self):
        """Returns an ASE-compatible NWChem calculator instance."""
        if self.calculator_type != "nwchem":
            raise ValueError("Invalid calculator_type. The only valid option is 'nwchem'.")

        return NWChem(
            theory=self.theory,
            xc=self.xc,
            basis=self.basis,
            kpts=self.kpts,
            directory=self.directory,
            command=self.command,
        )
