from pydantic import BaseModel, Field


class Psi4Calc(BaseModel):
    """PSI4 quantum chemistry calculator configuration.

    This class defines the configuration parameters for PSI4 quantum chemistry
    calculations. It supports various quantum chemical methods, basis sets, and
    SCF convergence parameters.

    Parameters
    ----------
    calculator_type : str, optional
        Type of calculator. Only 'psi4' is supported, by default 'psi4'
    method : str, optional
        Computational method to be used. Common methods include:
        ['hf', 'mp2', 'ccsd', 'ccsd(t)', 'df-mp2', 'b3lyp', 'pbe0', 'm06-2x'],
        by default 'b3lyp'
    basis : str, optional
        Basis set to be used. Common basis sets include:
        ['sto-3g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'def2-svp', 'aug-cc-pvdz'],
        by default '6-31g'
    reference : str, optional
        Wavefunction reference type. Options: 'rhf' (default), 'uhf', 'rohf',
        by default 'rhf'
    scf_type : str, optional
        SCF solver type. Options: 'pk' (default), 'df' (Density-Fitted),
        'cd' (Cholesky Decomposition), by default 'pk'
    maxiter : int, optional
        Maximum number of SCF iterations, by default 50
    """

    calculator_type: str = Field(
        default="psi4", description="Type of calculator. Only 'psi4' is supported."
    )
    method: str = Field(
        default="b3lyp",
        description=(
            "Computational method to be used. List of common methods: ['hf', 'mp2', 'ccsd', 'ccsd(t)', 'df-mp2', 'b3lyp', 'pbe0', 'm06-2x']"
        ),
    )
    basis: str = Field(
        default="6-31g",
        description=(
            "Basis set to be used. List of common basis set: ['sto-3g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'def2-svp', 'aug-cc-pvdz'] "
        ),
    )
    reference: str = Field(
        default="rhf",
        description="Wavefunction reference type. Options: 'rhf' (default), 'uhf', 'rohf'.",
    )

    scf_type: str = Field(
        default="pk",
        description="SCF solver type. Options: 'pk' (default), 'df' (Density-Fitted), 'cd' (Cholesky Decomposition).",
    )

    maxiter: int = Field(
        default=50, description="Maximum number of SCF iterations. Default is 50."
    )

    def get_calculator(self) -> dict:
        """Get a dictionary of PSI4 calculation parameters.

        Constructs and returns a dictionary containing the parameters
        for a PSI4 calculation based on the current settings.

        Returns
        -------
        dict
            A dictionary with PSI4 calculation parameters including method,
            basis, reference, SCF type, and maximum iterations
        """
        params = {
            "method": self.method,
            "basis": self.basis,
            "reference": self.reference,
            "scf_type": self.scf_type,
            "maxiter": self.maxiter,
        }
        return params
