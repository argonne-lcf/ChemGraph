from pydantic import BaseModel, Field


class EMTCalc(BaseModel):
    """Effective Medium Theory (EMT) calculator configuration.

    This class defines the configuration parameters for the EMT calculator,
    which is a simple empirical potential for metals. It provides a fast
    approximation for metallic systems.

    Parameters
    ----------
    calculator_type : str, optional
        Calculator type. Currently supports only 'emt', by default 'emt'
    asap_cutoff : bool, optional
        If True, the cutoff mimics how ASAP does it; the global cutoff is
        chosen from the largest atom present in the simulation, by default False

    Notes
    -----
    The EMT calculator is a simple empirical potential that works well for
    metallic systems. It is particularly useful for quick calculations and
    as a starting point for more accurate methods.
    """

    calculator_type: str = Field(
        default="emt", description="Calculator type. Currently supports only 'emt'."
    )
    asap_cutoff: bool = Field(
        default=False,
        description="If True, the cutoff mimics how ASAP does it; the global cutoff is chosen from the largest atom present in the simulation.",
    )

    def get_calculator(self):
        """Get an ASE-compatible EMT calculator instance.

        Returns
        -------
        EMT
            An ASE-compatible EMT calculator instance with the specified
            configuration parameters

        Raises
        ------
        ValueError
            If an invalid calculator_type is specified
        """
        if self.calculator_type != "emt":
            raise ValueError("Invalid calculator_type. The only valid option is 'emt'.")

        from ase.calculators.emt import EMT

        return EMT(asap_cutoff=self.asap_cutoff)
