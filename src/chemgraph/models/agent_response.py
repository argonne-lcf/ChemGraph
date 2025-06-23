from pydantic import BaseModel, Field
from typing import Union
from chemgraph.models.atomsdata import AtomsData


class VibrationalFrequency(BaseModel):
    """
    Schema for storing vibrational frequency results from a simulation.

    Attributes
    ----------
    frequency_cm1 : list[str]
        List of vibrational frequencies in inverse centimeters (cm⁻¹).
        Each entry is a string representation of the frequency value.
    """

    frequency_cm1: list[str] = Field(
        ...,
        description="List of vibrational frequencies in cm-1.",
    )


class ScalarResult(BaseModel):
    """
    Schema for storing a scalar numerical result from a simulation or calculation.

    Attributes
    ----------
    value : float
        The numerical value of the scalar result (e.g., 1.23).
    property : str
        The name of the physical or chemical property represented (e.g., 'enthalpy', 'Gibbs free energy').
    unit : str
        The unit associated with the result (e.g., 'eV', 'kJ/mol').
    """

    value: float = Field(..., description="Scalar numerical result like enthalpy")
    property: str = Field(
        ...,
        description="Name of the property, e.g. 'enthalpy', 'Gibbs free energy'",
    )
    unit: str = Field(..., description="Unit of the result, e.g. 'eV'")


class ResponseFormatter(BaseModel):
    """Defined structured output to the user."""

    answer: Union[
        str,
        ScalarResult,
        VibrationalFrequency,
        AtomsData,
        list[AtomsData],
        list[VibrationalFrequency],
        list[ScalarResult],
    ] = Field(
        description=(
            "Structured answer to the user's query. Use:\n"
            "- `str` for general or explanatory responses or SMILES string.\n"
            "- `VibrationalFrequency` for vibrational frequecies.\n"
            "- `ScalarResult` for single numerical properties (e.g. enthalpy).\n"
            "- `AtomsData` for atomic geometries (XYZ coordinate, etc.) and optimized structures."
        )
    )
