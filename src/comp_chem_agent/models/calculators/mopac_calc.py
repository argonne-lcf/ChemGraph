# Keywords and parameters obtained from QCEngine: https://github.com/MolSSI/QCEngine
# MOPAC parameters for CompChemAgent

from pydantic import BaseModel, Field

class MopacCalc(BaseModel):
    calculator_type: str = Field(
        default="mopac", description="Type of calculator. Currently supports only 'mopac'."
    )
    method: str = Field(
        default="am1", description="Computational method to be used. Available methods include ['mndo', 'am1', 'pm3', 'rm1', 'mndod', 'pm6', 'pm6-d3', 'pm6-dh+', 'pm6-dh2', 'pm6-dh2x', 'pm6-d3h4', 'pm6-3dh4x', 'pm7', 'pm7-ts']"
    )
    iter: int = Field(
        default=100, description="Maximum number of self-consistent field (SCF) iterations allowed."
    )
    pulay: bool = Field(
        default=True, description="Enable Pulay's convergence acceleration for the SCF procedure."
    )

    def get_calculator(self):
        """Returns the parameters for MOPAC with the specified parameters."""
        return {
            "method": self.method,
            "ITER": self.iter,
            "PULAY": self.pulay,
        }
