"""Engine-neutral schemas for adsorption simulations."""

from __future__ import annotations

import math
import re
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator

CanonicalAdsorbate: TypeAlias = Literal["CO2", "N2", "H2O"]
EngineOptionScalar: TypeAlias = bool | int | float | str
EngineOptionValue: TypeAlias = EngineOptionScalar | list[EngineOptionScalar]

_OPTION_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MANAGED_OPTIONS = {
    "component",
    "cutoffcoulomb",
    "cutoffvdw",
    "frameworkname",
    "moleculeName".lower(),
    "numberofinitializationcycles",
    "numberofproductioncycles",
    "pressure",
    "temperature",
    "unitcells",
}


def _validate_engine_options(options: dict[str, EngineOptionValue]) -> None:
    for key, value in options.items():
        if not _OPTION_KEY.fullmatch(key):
            raise ValueError(f"Unsafe engine option name: {key!r}")
        if key.lower() in _MANAGED_OPTIONS:
            raise ValueError(f"Engine option {key!r} is managed by ChemGraph")
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ValueError(f"Engine option {key!r} cannot be an empty list")
        for item in values:
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"Engine option {key!r} must be finite")
            if isinstance(item, str) and ("\n" in item or "\r" in item):
                raise ValueError(f"Engine option {key!r} cannot contain newlines")


class AdsorptionComponent(BaseModel):
    """One adsorbate in a pure-gas or mixture simulation."""

    name: CanonicalAdsorbate
    mole_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    ideal_gas_rosenbluth_weight: PositiveFloat = 1.0
    fugacity_coefficient: PositiveFloat | Literal["PR-EOS"] | None = None
    translation_probability: float = Field(default=1.0, ge=0.0)
    rotation_probability: float = Field(default=1.0, ge=0.0)
    reinsertion_probability: float = Field(default=1.0, ge=0.0)
    identity_change_probability: float | None = Field(default=None, ge=0.0)
    swap_probability: float = Field(default=2.0, ge=0.0)
    create_number_of_molecules: int = Field(default=0, ge=0)
    engine_options: dict[str, EngineOptionValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_options(self) -> AdsorptionComponent:
        _validate_engine_options(self.engine_options)
        return self


def _validate_components(
    components: list[AdsorptionComponent],
) -> list[AdsorptionComponent]:
    names = [component.name for component in components]
    if len(names) != len(set(names)):
        raise ValueError("Adsorption component names must be unique")

    if len(components) == 1:
        component = components[0]
        if component.mole_fraction is None:
            component.mole_fraction = 1.0
        elif not math.isclose(component.mole_fraction, 1.0, abs_tol=1e-6):
            raise ValueError("A pure-gas mole fraction must equal 1.0")
        return components

    if any(component.mole_fraction is None for component in components):
        raise ValueError("Every mixture component requires mole_fraction")
    total = sum(float(component.mole_fraction) for component in components)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Mixture mole fractions must sum to 1.0; got {total:g}")
    return components


class AdsorptionCondition(BaseModel):
    """Temperature and pressure for one adsorption state point."""

    temperature: PositiveFloat = 298.15
    pressure: PositiveFloat = 101325.0


class AdsorptionRequest(BaseModel):
    """A single structure, condition, and gas composition."""

    input_structure_file: str = Field(min_length=1)
    output_result_file: str = Field(default="raspa.log", min_length=1)
    temperature: PositiveFloat = 298.15
    pressure: PositiveFloat = 101325.0
    n_cycles: PositiveInt = 10000
    cutoff: PositiveFloat = 12.8
    components: list[AdsorptionComponent] = Field(min_length=1, max_length=3)
    engine_options: dict[str, EngineOptionValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> AdsorptionRequest:
        self.components = _validate_components(self.components)
        _validate_engine_options(self.engine_options)
        return self


class AdsorptionEnsembleRequest(BaseModel):
    """A structure ensemble evaluated at explicit state points."""

    input_structures: str | list[str] = ""
    remote_structure_directory: str | None = None
    remote_structure_files: list[str] | None = None
    output_result_file: str = Field(default="raspa.log", min_length=1)
    conditions: list[AdsorptionCondition] = Field(
        default_factory=lambda: [AdsorptionCondition()], min_length=1
    )
    n_cycles: PositiveInt = 10000
    cutoff: PositiveFloat = 12.8
    components: list[AdsorptionComponent] = Field(min_length=1, max_length=3)
    engine_options: dict[str, EngineOptionValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> AdsorptionEnsembleRequest:
        self.components = _validate_components(self.components)
        _validate_engine_options(self.engine_options)
        remote_modes = bool(self.remote_structure_directory) + bool(
            self.remote_structure_files
        )
        if remote_modes > 1:
            raise ValueError(
                "Use remote_structure_directory or remote_structure_files, not both"
            )
        if not self.input_structures and not remote_modes:
            raise ValueError(
                "Provide input_structures, remote_structure_directory, or "
                "remote_structure_files"
            )
        if self.input_structures and remote_modes:
            raise ValueError("Local and remote structure inputs are mutually exclusive")
        return self


class ComponentUptake(BaseModel):
    name: CanonicalAdsorbate
    feed_mole_fraction: float
    uptake: float
    uncertainty: float | None = None
    unit: Literal["mol/kg"] = "mol/kg"


class AdsorptionSelectivity(BaseModel):
    numerator: CanonicalAdsorbate
    denominator: CanonicalAdsorbate
    value: float | None
    message: str | None = None


class AdsorptionResult(BaseModel):
    status: Literal["success", "failure"]
    engine: str
    temperature: float
    pressure: float
    components: list[ComponentUptake] = Field(default_factory=list)
    selectivities: list[AdsorptionSelectivity] = Field(default_factory=list)
    cif_path: str
    working_directory: str
    stdout_path: str
    stderr_path: str
    return_code: int | None = None
    wall_time_seconds: float | None = None
    message: str | None = None
