# Keywords and parameters obtained from QCEngine: https://github.com/MolSSI/QCEngine
# MOPAC parameters for CompChemAgent

from pydantic import BaseModel, Field


class MopacCalc(BaseModel):
    """MOPAC calculator configuration.

    This class defines the configuration parameters for the MOPAC calculator,
    which is a semi-empirical quantum chemistry program. It provides various
    semi-empirical methods for molecular calculations.

    Parameters
    ----------
    calculator_type : str, optional
        Type of calculator. Currently supports only 'mopac', by default 'mopac'
    method : str, optional
        Computational method to be used. Available methods include:
        ['mndo', 'am1', 'pm3', 'rm1', 'mndod', 'pm6', 'pm6-d3', 'pm6-dh+',
        'pm6-dh2', 'pm6-dh2x', 'pm6-d3h4', 'pm6-3dh4x', 'pm7', 'pm7-ts'],
        by default 'am1'
    iter : int, optional
        Maximum number of self-consistent field (SCF) iterations allowed,
        by default 100
    pulay : bool, optional
        Enable Pulay's convergence acceleration for the SCF procedure,
        by default True

    Notes
    -----
    MOPAC is a semi-empirical quantum chemistry program that provides
    various methods for molecular calculations. The available methods
    range from basic semi-empirical methods (MNDO, AM1, PM3) to more
    advanced ones with dispersion corrections (PM6-D3, PM6-DH+).
    """

    calculator_type: str = Field(
        default="mopac",
        description="Type of calculator. Currently supports only 'mopac'.",
    )
    method: str = Field(
        default="am1",
        description="Computational method to be used. Available methods include ['mndo', 'am1', 'pm3', 'rm1', 'mndod', 'pm6', 'pm6-d3', 'pm6-dh+', 'pm6-dh2', 'pm6-dh2x', 'pm6-d3h4', 'pm6-3dh4x', 'pm7', 'pm7-ts']",
    )
    iter: int = Field(
        default=100,
        description="Maximum number of self-consistent field (SCF) iterations allowed.",
    )
    pulay: bool = Field(
        default=True,
        description="Enable Pulay's convergence acceleration for the SCF procedure.",
    )

    def get_calculator(self):
        """Get MOPAC calculator parameters.

        Returns
        -------
        dict
            A dictionary containing the MOPAC calculator parameters:
            - method: The computational method to use
            - ITER: Maximum number of SCF iterations
            - PULAY: Whether to use Pulay's convergence acceleration
        """
        return {
            "method": self.method,
            "ITER": self.iter,
            "PULAY": self.pulay,
        }
