"""Validated inputs for the supported H2O gRASPA-SYCL workflow."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SimulationCondition(BaseModel):
    temperature: float = Field(
        default=298.15,
        gt=0,
        allow_inf_nan=False,
        description="Temperature in Kelvin (K).",
    )
    pressure: float = Field(
        default=101325.0,
        ge=0,
        allow_inf_nan=False,
        description="Pressure in Pascal (Pa).",
    )


class _GraspaOptions(BaseModel):
    output_result_file: str = Field(
        default="raspa.log",
        min_length=1,
        description="Stdout filename inside the unique simulation directory.",
    )
    n_cycles: int = Field(
        default=10000,
        gt=0,
        description="Monte Carlo cycles per initialization/production phase.",
    )
    adsorbate: Literal["H2O"] = Field(description="Supported adsorbate: H2O.")

    @field_validator("output_result_file")
    @classmethod
    def safe_output_name(cls, value: str) -> str:
        name = Path(value).name.lower()
        if (
            not name
            or name
            in {
                ".",
                "..",
                "simulation.input",
                "simulation.input.tmp",
                "raspa.err",
                "results.json",
                "results.json.tmp",
            }
            or Path(name).suffix.lower() in {".cif", ".def"}
        ):
            raise ValueError(
                "Output filename must not overwrite simulation inputs or metadata"
            )
        return value


class graspa_input_schema(_GraspaOptions, SimulationCondition):
    input_structure_file: str = Field(
        min_length=1, description="Path to the input CIF file."
    )
    output_directory: str | None = Field(
        default=None,
        min_length=1,
        description="Worker-side root for unique run directories; defaults to graspa_runs under CHEMGRAPH_LOG_DIR or cwd.",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Optional simulation time limit in seconds; no limit by default.",
    )

    @model_validator(mode="after")
    def unambiguous_output_root(self):
        if self.output_directory is not None and Path(
            self.output_result_file
        ).parent != Path("."):
            raise ValueError(
                "Use output_directory with a bare output_result_file filename"
            )
        return self


class graspa_input_schema_ensemble(_GraspaOptions):
    input_structures: str | list[str] = Field(
        default="",
        description="Local directory of CIF files or a nonempty list of CIF paths on a shared filesystem.",
    )
    remote_structure_directory: str | None = Field(
        default=None,
        min_length=1,
        description="Pre-staged CIF directory on the worker filesystem; exclusive with input_structures.",
    )
    conditions: list[SimulationCondition] = Field(
        default_factory=lambda: [SimulationCondition()],
        min_length=1,
        description="Temperature/pressure conditions for every structure.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_deferred_controls(cls, values):
        if isinstance(values, Mapping):
            unsupported = {
                "output_directory", "timeout_seconds", "discovery_timeout_seconds"
            }.intersection(values)
            if unsupported:
                raise ValueError(
                    f"Unsupported ensemble controls: {', '.join(sorted(unsupported))}. "
                    "Omit these fields until ensemble support is implemented."
                )
        return values

    @model_validator(mode="after")
    def one_input_source(self):
        if bool(self.input_structures) == bool(self.remote_structure_directory):
            raise ValueError(
                "Provide exactly one of input_structures or remote_structure_directory"
            )
        if isinstance(self.input_structures, list) and any(
            not p.strip() for p in self.input_structures
        ):
            raise ValueError("Input file paths must be nonempty")
        return self
