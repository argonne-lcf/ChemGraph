from pydantic import BaseModel, Field


class graspa_input_schema(BaseModel):
    input_structure_file: str = Field(
        description="Path to the input CIF file containing the atomic structure for the simulation."
    )
    output_result_file: str = Field(
        default="raspa.log",
        description="Name of a JSON file where simulation results will be saved.",
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
    adsorbates: list[str] = Field(
        description="List of adsorbates for the simulations. Supported adsorbates are CO2, N2 and H2O",
    )
    adsorbate_compositions: list[float] = Field(
        description="List of adsorbate compositions in the same order as list of adsorbates."
    )


class graspa_input_schema_ensemble(BaseModel):
    input_structure_directory: str = Field(
        description="Path to a folder of input structures containing the atomic structure for the simulations."
    )
    output_result_file: str = Field(
        default="raspa.log",
        description="Name of a JSON file where simulation results will be saved.",
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
    adsorbates: list[str] = Field(
        description="List of adsorbates for the simulations. Supported adsorbates are CO2, N2 and H2O",
    )
    adsorbate_compositions: list[float] = Field(
        description="List of adsorbate compositions in the same order as list of adsorbates."
    )
