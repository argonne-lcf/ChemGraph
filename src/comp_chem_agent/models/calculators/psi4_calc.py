from pydantic import BaseModel, Field

class Psi4Calc(BaseModel):
    calculator_type: str = Field(
        default="psi4",
        description="Type of calculator. Only 'psi4' is supported."
    )
    method: str = Field(
        default="b3lyp",
        description=(
            "Computational method to be used. List of common methods: ['hf', 'mp2', 'ccsd', 'ccsd(t)', 'df-mp2', 'b3lyp', 'pbe0', 'm06-2x']"
        )
    )
    basis: str = Field(
        default="6-31g",
        description=(
            "Basis set to be used. List of common basis set: ['sto-3g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'def2-svp', 'aug-cc-pvdz'] "
        )
    )
    reference: str = Field(
        default="rhf",
        description="Wavefunction reference type. Options: 'rhf' (default), 'uhf', 'rohf'."
    )

    scf_type: str = Field(
        default="pk",
        description="SCF solver type. Options: 'pk' (default), 'df' (Density-Fitted), 'cd' (Cholesky Decomposition)."
    )

    maxiter: int = Field(
        default=50,
        description="Maximum number of SCF iterations. Default is 50."
    )

    def get_calculator(self) -> dict:
        """
        Constructs and returns a dictionary containing the parameters
        for a Psi4 calculation based on the current settings.
        
        Returns:
            dict: A dictionary with Psi4 calculation parameters.
        """
        params = {
            "method": self.method,
            "basis": self.basis,
            "reference": self.reference,
            "scf_type": self.scf_type,
            "maxiter": self.maxiter,
        }
        return params
