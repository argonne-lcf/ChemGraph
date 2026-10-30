# Keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/orca.html#ORCA
# Orca parameters for CompChemAgent

from typing import Optional
from pydantic import BaseModel, Field
from ase.calculators.orca import ORCA, OrcaProfile
import logging
import shlex
from pydantic.json_schema import SkipJsonSchema
from chemgraph.utils.orca import resolve_orca


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
        default=1,
        ge=1,
        description="Spin multiplicity (2S+1): 1=singlet, 2=doublet, etc.",
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
    profile: SkipJsonSchema[Optional[OrcaProfile]] = Field(
        default=None, exclude=True, description="Optional Python-only ORCA profile configuration."
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

        profile = self.profile
        if profile is None:
            try:
                executable = resolve_orca()
            except FileNotFoundError as exc:
                from ase.calculators.calculator import BadConfiguration
                raise BadConfiguration(str(exc)) from exc
            logging.getLogger(__name__).debug("ORCA executable: %s", executable)
            profile = OrcaProfile(command=shlex.quote(executable))

        return ORCA(
            charge=self.charge,
            mult=self.multiplicity,
            orcasimpleinput=self.orcasimpleinput,
            orcablocks=self.orcablocks,
            directory=self.directory,
            profile=profile,
        )

    def get_multiplicity(self) -> Optional[int]:
        """Return spin multiplicity (2S+1) for thermochemistry."""
        return self.multiplicity
