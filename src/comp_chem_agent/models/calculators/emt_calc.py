from pydantic import BaseModel, Field


class EMTCalc(BaseModel):
    calculator_type: str = Field(
        default="emt", description="Calculator type. Currently supports only 'emt'."
    )
    asap_cutoff: bool = Field(
        default=False,
        description="If True, the cutoff mimics how ASAP does it; the global cutoff is chosen from the largest atom present in the simulation.",
    )

    def get_calculator(self):
        """Returns the ASE EMT calculator with the specified parameters."""
        if self.calculator_type != "emt":
            raise ValueError("Invalid calculator_type. The only valid option is 'emt'.")

        from ase.calculators.emt import EMT

        return EMT(asap_cutoff=self.asap_cutoff)
