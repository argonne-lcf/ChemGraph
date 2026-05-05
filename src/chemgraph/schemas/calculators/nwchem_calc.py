# Main keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/nwchem.html
# Parameters for NWChem calculator in CompChemAgent

from typing import Optional, Union, Dict
from pydantic import BaseModel, Field
from ase.calculators.nwchem import NWChem


class NWChemCalc(BaseModel):
    """NWChem quantum chemistry calculator configuration.

    This class defines the configuration parameters for NWChem quantum chemistry
    calculations. It supports various quantum chemical methods, basis sets, and
    periodic calculations through the NWChem program.

    Parameters
    ----------
    calculator_type : str, optional
        Calculator type. Currently supports only 'nwchem', by default 'nwchem'
    theory : str, optional
        NWChem module to be used. Options: 'dft', 'scf', 'mp2', 'ccsd', 'tce',
        'tddft', 'pspw', 'band', 'paw', by default 'dft'
    xc : str, optional
        Exchange-correlation functional (only applicable for DFT calculations),
        by default 'PBE'
    basis : str or dict, optional
        Basis set to use. Can be a string for all elements or a dictionary
        mapping elements to basis sets, by default '6-31G'
    kpts : tuple or dict, optional
        K-point mesh for periodic calculations, by default None
    directory : str, optional
        Working directory for NWChem calculations, by default '.'
    command : str, optional
        Command to execute NWChem (e.g., 'nwchem PREFIX.nwi > PREFIX.nwo'),
        by default None
    charge : int, optional
        Total charge of the system, by default None
    multiplicity : int, optional
        Spin multiplicity (2S+1) of the system, by default None.
        For molecular theories ('dft', 'scf', 'mp2', 'ccsd', 'tce', 'tddft') this is
        injected into the theory block as ``mult``. For 'scf', NWChem expects
        ``nopen`` (number of unpaired electrons); set ``scf={'nopen': N}`` manually
        if you need finer control.
    """

    calculator_type: str = Field(
        default="nwchem",
        description="Calculator type. Currently supports only 'nwchem'.",
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
    directory: str = Field(
        default=".", description="Working directory for NWChem calculations."
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to execute NWChem (e.g., 'nwchem PREFIX.nwi > PREFIX.nwo').",
    )
    charge: Optional[int] = Field(
        default=None, description="Total charge of the system."
    )
    multiplicity: Optional[int] = Field(
        default=None,
        description=(
            "Spin multiplicity (2S+1). Injected into the theory block as 'mult' "
            "for dft/mp2/ccsd/tce/tddft; for 'scf' theory NWChem expects 'nopen' "
            "(unpaired electrons) which is not auto-set here."
        ),
        ge=1,
    )

    def get_calculator(self):
        """Get an ASE-compatible NWChem calculator instance.

        Returns
        -------
        NWChem
            An ASE-compatible NWChem calculator instance

        Raises
        ------
        ValueError
            If an invalid calculator_type is specified
        """
        if self.calculator_type != "nwchem":
            raise ValueError(
                "Invalid calculator_type. The only valid option is 'nwchem'."
            )

        kwargs = dict(
            theory=self.theory,
            xc=self.xc,
            basis=self.basis,
            kpts=self.kpts,
            directory=self.directory,
            command=self.command,
        )

        # NWChem accepts charge/multiplicity inside the theory-specific block.
        block: Dict[str, Union[int, str]] = {}
        if self.charge is not None:
            block["charge"] = self.charge
        if self.multiplicity is not None and self.theory != "scf":
            block["mult"] = self.multiplicity
        if block and self.theory in {"dft", "mp2", "ccsd", "tce", "tddft"}:
            kwargs[self.theory] = block

        return NWChem(**kwargs)

    def get_multiplicity(self) -> Optional[int]:
        """Return spin multiplicity (2S+1) for thermochemistry."""
        return self.multiplicity
