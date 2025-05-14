# Keywords obtained from https://github.com/tblite/tblite/blob/main/python/tblite/ase.py
# TBLite calculator parameters for CompChemAgent
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

try:
    from tblite.ase import TBLite
except ImportError:
    logging.warning(
        "tblite is not installed. If you want to use tblite, please install it using 'pip install tblite'."
    )


class TBLiteCalc(BaseModel):
    calculator_type: str = Field(
        default="TBLite",
        description="Calculator type for XTB methods. Only supports TBLite",
    )
    method: str = Field(
        default="GFN2-xTB",
        description="Underlying method for energy and forces. Options are GFN2-xTB and GFN1-xTB.",
    )
    charge: Optional[float] = Field(default=None, description="Total charge of the system")
    multiplicity: Optional[int] = Field(
        default=None, description="Total multiplicity of the system"
    )
    accuracy: float = Field(default=1.0, description="Numerical accuracy of the calculation")
    electronic_temperature: float = Field(
        default=300.0, description="Electronic temperature in Kelvin"
    )
    max_iterations: int = Field(
        default=250, description="Iterations for self-consistent evaluation"
    )
    initial_guess: str = Field(
        default="sad", description="Initial guess for wavefunction (sad or eeq)"
    )
    mixer_damping: float = Field(
        default=0.4, description="Damping parameter for self-consistent mixer"
    )
    electric_field: Optional[Optional[List[float]]] = Field(
        default=None, description="Uniform electric field vector (in V/A)"
    )
    spin_polarization: Optional[float] = Field(
        default=None, description="Spin polarization (scaling factor)"
    )
    cache_api: bool = Field(default=True, description="Reuse generated API objects (recommended)")
    verbosity: int = Field(default=0, description="Set verbosity of printout")

    def get_calculator(self):
        """Returns an ASE-compatible TBLite calculator instance."""
        return TBLite(
            method=self.method,
            charge=self.charge,
            multiplicity=self.multiplicity,
            accuracy=self.accuracy,
            electronic_temperature=self.electronic_temperature,
            max_iterations=self.max_iterations,
            initial_guess=self.initial_guess,
            mixer_damping=self.mixer_damping,
            electric_field=self.electric_field,
            spin_polarization=self.spin_polarization,
            cache_api=self.cache_api,
            verbosity=self.verbosity,
        )
