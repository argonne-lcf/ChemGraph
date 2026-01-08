# The gRASPA schema is configured to work within the capability of the SYCL version. Further modifications are needed to
# make it compatible with gRASPA-CUDA.
from typing import Union
from pydantic import BaseModel, Field


class SimulationCondition(BaseModel):
    """
    Helper model to group temperature and pressure for a single simulation state.
    """

    temperature: float = Field(
        default=298.15,
        description="Temperature in Kelvin (K).",
    )
    pressure: float = Field(
        default=101325.0,
        description="Pressure in Pascal (Pa).",
    )


class graspa_input_schema(BaseModel):
    input_structure_file: str = Field(
        description="Path to the input CIF file containing the atomic structure for the simulation."
    )
    output_result_file: str = Field(
        default="raspa.log",
        description="Name of a file where simulation results will be saved.",
    )
    temperature: float = Field(
        default=298.15,
        description="Temperature in Kelvin (K).",
    )
    pressure: float = Field(
        default=101325.0,
        description="Pressure in Pascal (Pa).",
    )
    n_cycles: int = Field(
        default=10000,
        description="Number of Monte Carlo cycles",
    )
    adsorbate: str = Field(
        description="Adsorbate name for the simulations. Supported adsorbate is 'H2O'",
    )


class graspa_input_schema_ensemble(BaseModel):
    input_structures: Union[str, list[str]] = Field(
        description="Path to a directory of CIF files OR a specific list of file paths."
    )
    output_result_file: str = Field(
        default="raspa.log",
        description="Name of a file where each simulation results will be saved.",
    )
    conditions: list[SimulationCondition] = Field(
        default_factory=lambda: [SimulationCondition()],
        description="List of temperature (K) and pressure (Pa) conditions to simulate.",
    )
    n_cycles: int = Field(
        default=10000,
        description="Number of Monte Carlo cycles",
    )
    adsorbate: str = Field(
        description="Adsorbate name for the simulations. Supported adsorbate is 'H2O'",
    )
