"""Compatibility schemas for the original gRASPA tools."""

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from chemgraph.schemas.adsorption_schema import (
    AdsorptionComponent,
    AdsorptionCondition,
    AdsorptionEnsembleRequest,
    AdsorptionRequest,
    CanonicalAdsorbate,
)


class SimulationCondition(AdsorptionCondition):
    """
    Helper model to group temperature and pressure for a single simulation state.
    """

class graspa_input_schema(BaseModel):
    input_structure_file: str = Field(
        description="Path to the input CIF file containing the atomic structure for the simulation."
    )
    output_result_file: str = Field(
        default="raspa.log",
        description="Name of a file where simulation results will be saved.",
    )
    temperature: PositiveFloat = Field(
        default=298.15,
        description="Temperature in Kelvin (K).",
    )
    pressure: PositiveFloat = Field(
        default=101325.0,
        description="Pressure in Pascal (Pa).",
    )
    n_cycles: PositiveInt = Field(
        default=10000,
        description="Number of Monte Carlo cycles",
    )
    cutoff: PositiveFloat = Field(default=12.8)
    adsorbate: CanonicalAdsorbate = Field(
        description="Adsorbate name: CO2, N2, or H2O.",
    )

    def to_adsorption_request(self) -> AdsorptionRequest:
        return AdsorptionRequest(
            input_structure_file=self.input_structure_file,
            output_result_file=self.output_result_file,
            temperature=self.temperature,
            pressure=self.pressure,
            n_cycles=self.n_cycles,
            cutoff=self.cutoff,
            components=[AdsorptionComponent(name=self.adsorbate)],
        )


class graspa_input_schema_ensemble(BaseModel):
    input_structures: str | list[str] = Field(
        default="",
        description="Path to a directory of CIF files OR a specific list of file paths. Required unless remote_structure_directory is provided.",
    )
    remote_structure_directory: str | None = Field(
        default=None,
        description=(
            "Path to pre-staged CIF files on the remote HPC filesystem. "
            "When provided, workers read structures directly from this path. "
            "Use the transfer_files tool to stage files first."
        ),
    )
    remote_structure_files: list[str] | None = Field(
        default=None,
        description="Explicit pre-staged remote CIF paths.",
    )
    output_result_file: str = Field(
        default="raspa.log",
        description="Name of a file where each simulation results will be saved.",
    )
    conditions: list[SimulationCondition] = Field(
        default_factory=lambda: [SimulationCondition()],
        description="List of temperature (K) and pressure (Pa) conditions to simulate.",
    )
    n_cycles: PositiveInt = Field(
        default=10000,
        description="Number of Monte Carlo cycles",
    )
    cutoff: PositiveFloat = Field(default=12.8)
    adsorbate: CanonicalAdsorbate = Field(
        description="Adsorbate name: CO2, N2, or H2O.",
    )

    def to_adsorption_request(self) -> AdsorptionEnsembleRequest:
        return AdsorptionEnsembleRequest(
            input_structures=self.input_structures,
            remote_structure_directory=self.remote_structure_directory,
            remote_structure_files=self.remote_structure_files,
            output_result_file=self.output_result_file,
            conditions=[condition.model_dump() for condition in self.conditions],
            n_cycles=self.n_cycles,
            cutoff=self.cutoff,
            components=[AdsorptionComponent(name=self.adsorbate)],
        )
