from pydantic import BaseModel, Field


class GRASPAInputSchema(BaseModel):
    output_path: str = Field(
        description="Absolute or relative path to the directory where gRASPA output files will be stored. If not provided, the output will be stored in the current working directory."
    )
    cif_path: str = Field(
        description="Absolute or relative path to the directory where the CIF file is stored."
    )
    mof_name: str = Field(description="Name of the MOF excluding .cif extension")
    adsorbate: str = Field(
        default='CO2', description="Name of the adsorbate molecule. Only support CO2, H2, CH4 and N2."
    )
    temperature: float = Field(default=300, description="Simulation temperature in Kelvin.")
    pressure: float = Field(default=1e5, description="Simulation pressure in Pascal.")
    n_cycle: int = Field(
        default=100, description="Number of Monte Carlo steps to run in the GCMC simulation."
    )
    cutoff: float = Field(default=12.8, description="The LJ and Coulomb cutoff in Angstrom")
    graspa_cmd: str= Field(
        default="/eagle/projects/HPCBot/thang/soft/gRASPA/src_clean/nvc_main.x >> raspa.log",
        description="The command to run gRASPA. If not provided, the default command will be used."
    )
    graspa_version: str = Field(
        default="cuda",
        description="The version of gRASPA to use. Only support 'cuda' and 'sycl'."
    )