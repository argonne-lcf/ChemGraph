from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json


class Component(BaseModel):
    molecule_name: str = Field(..., alias="MoleculeName")
    molecule_definition: str = Field(default="Local", alias="MoleculeDefinition")
    translation_probability: float = Field(default=1.0, alias="TranslationProbability")
    rotation_probability: float = Field(default=1.0, alias="RotationProbability")
    reinsertion_probability: float = Field(default=1.0, alias="ReinsertionProbability")
    swap_probability: float = Field(default=2.0, alias="SwapProbability")
    create_number_of_molecules: int = Field(default=0, alias="CreateNumberOfMolecules")


class SimulationInput(BaseModel):
    simulation_type: str = Field(default="MonteCarlo", alias="SimulationType")
    number_of_cycles: int = Field(default=100, alias="NumberOfCycles")
    number_of_initialization_cycles: int = Field(
        default=100, alias="NumberOfInitializationCycles"
    )
    print_every: int = Field(default=100, alias="PrintEvery")
    print_forcefield_to_output: str = Field(
        default="no", alias="PrintForcefieldToOutput"
    )
    movies: str = Field(default="yes", alias="Movies")
    write_movies_every: int = Field(default=100, alias="WriteMoviesEvery")
    cutoff: float = Field(default=12.8, alias="Cutoff")
    forcefield: str = Field(default="Local", alias="Forcefield")
    charge_method: str = Field(default="Ewald", alias="ChargeMethod")
    ewald_precision: float = Field(default=1e-6, alias="EwaldPrecision")
    cutoff_coulomb: float = Field(default=12.8, alias="CutoffCoulomb")
    use_charges_from_cif_file: str = Field(default="Yes", alias="UseChargesFromCIFFile")
    framework: int = Field(
        default=0, alias="Framework", description="Index of framework, starts from 0."
    )
    framework_name: str = Field(default=None, alias="FrameworkName")
    unit_cells: List[int] = Field(
        default=[1, 1, 1], alias="UnitCells", description="Framework name"
    )
    external_temperature: float = Field(
        default=273,
        alias="ExternalTemperature",
        description="Temperature in Kelvin (K)",
    )
    external_pressure: float = Field(
        default=1e6, alias="ExternalPressure", description="Pressure in Pascal (Pa)"
    )
    components: List[Component] = Field(..., alias="Component")

    class Config:
        populate_by_name = True  # Allows using both alias and field names

    def write_to_file(self, filename: str):
        with open(filename, "w") as f:
            data = self.model_dump(by_alias=True)  # Use aliases for keys
            for key, value in data.items():
                if isinstance(value, list) or isinstance(value, tuple):
                    if key == "Component":
                        for i, item in enumerate(value, start=0):
                            f.write(f"{key} {i}\n")
                            for sub_key, sub_value in item.items():
                                f.write(f"{sub_key} {sub_value}\n")
                    else:
                        f.write(f"{key} {value[0]} {value[1]} {value[2]}")
                else:
                    f.write(f"{key} {value}\n")


class ForceField(BaseModel):
    pass


class Mod(BaseModel):
    pass


class SimulationOutput(BaseModel):
    pass
