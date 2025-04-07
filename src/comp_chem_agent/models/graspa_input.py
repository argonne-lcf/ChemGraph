from pydantic import BaseModel, Field


class GRASPAInputSchema(BaseModel):
    output_path: str = Field(
        description="Absolute or relative path to the directory where gRASPA output files will be stored."
    )
    ddec6_path: str = Field(
        description="Absolute or relative path to the directory where DDEC6 output files are stored."
    )
    name: str = Field(description="Name of the MOF excluding .cif extension")
    adsorbate: str = Field(
        default='CO2', description="Name of the adsorbate molecule (e.g., 'CO2', 'CH4', 'N2')."
    )
    temperature: float = Field(default=300, description="Simulation temperature in Kelvin (K).")
    pressure: float = Field(default=1e4, description="Simulation pressure in Pascal.")
    n_cycle: int = Field(
        default=100, description="Number of Monte Carlo steps to run in the GCMC simulation."
    )
    cutoff: float = Field(default=12.8, description="The LJ and Coulomb cutoff in Angstrom")
