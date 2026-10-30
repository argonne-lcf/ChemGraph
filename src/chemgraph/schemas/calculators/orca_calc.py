# Keywords and parameters obtained from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/orca.html#ORCA
# Orca parameters for CompChemAgent

from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from ase.calculators.orca import ORCA, OrcaProfile
import logging
import shlex
from pydantic.json_schema import SkipJsonSchema
from chemgraph.utils.orca import resolve_orca


class DFTSettings(BaseModel):
    """Supported gas-phase closed-shell DFT settings; no native input directives."""

    model_config = ConfigDict(extra="forbid")
    functional: Literal["PBE", "PBE0", "B3LYP", "BP86"]
    basis: Literal["STO-3G", "def2-SVP", "def2-TZVP", "def2-TZVPD"]

    @field_validator("functional", "basis", mode="before")
    @classmethod
    def canonicalize(cls, value):
        choices = (
            "PBE",
            "PBE0",
            "B3LYP",
            "BP86",
            "STO-3G",
            "def2-SVP",
            "def2-TZVP",
            "def2-TZVPD",
        )
        return next(
            (
                v
                for v in choices
                if isinstance(value, str) and v.lower() == value.lower()
            ),
            value,
        )


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

    dft_settings: DFTSettings | None = None

    calculator_type: str = Field(
        default="orca", description="Calculator type. Currently supports only 'orca'."
    )
    charge: int = Field(default=0, description="Total charge of the system.")
    multiplicity: int = Field(
        default=1,
        ge=1,
        description="Spin multiplicity (2S+1): 1=singlet, 2=doublet, etc.",
    )
    orcasimpleinput: Optional[str] = Field(
        default=None,
        description="ORCA input keywords specifying method and basis set.",
    )
    orcablocks: Optional[str] = Field(
        default=None, description="Additional ORCA block settings."
    )
    directory: str = Field(
        default=".", description="Working directory for ORCA calculations."
    )
    profile: SkipJsonSchema[Optional[OrcaProfile]] = Field(
        default=None,
        exclude=True,
        description="Optional Python-only ORCA profile configuration.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_structured_unknowns(cls, value):
        if isinstance(value, dict) and value.get("dft_settings") is not None:
            unknown = set(value) - set(cls.model_fields)
            if unknown:
                raise ValueError(f"Unknown structured ORCA settings: {sorted(unknown)}")
        return value

    @model_validator(mode="after")
    def structured_contract(self):
        if self.dft_settings is not None:
            if (
                not {"charge", "multiplicity"} <= self.model_fields_set
                or self.multiplicity != 1
            ):
                raise ValueError(
                    "Structured ORCA requires explicit charge and multiplicity=1"
                )
            if (
                self.orcasimpleinput is not None
                or self.orcablocks is not None
                or self.profile is not None
                or self.directory != "."
            ):
                raise ValueError(
                    "Structured ORCA forbids raw input, profiles and non-default directories"
                )
        return self

    def get_calculator(self, *, driver=None, directory=None, budget=None):
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

        if self.dft_settings is not None:
            from chemgraph.tools.orca_execution import structured_calculator

            self.structured_contract()
            if driver is None or directory is None or budget is None:
                raise ValueError("Structured ORCA must run through run_ase_core")
            return structured_calculator(self, driver, directory, budget)

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
            orcasimpleinput=self.orcasimpleinput
            if self.orcasimpleinput is not None
            else "B3LYP def2-TZVP",
            orcablocks=self.orcablocks
            if self.orcablocks is not None
            else "%pal nprocs 1 end",
            directory=self.directory,
            profile=profile,
        )

    def get_multiplicity(self) -> Optional[int]:
        """Return spin multiplicity (2S+1) for thermochemistry."""
        return self.multiplicity
