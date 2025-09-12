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

class IRSpectrum(BaseModel):
    """
    Schema for storing vibrational frequency  and intensities from a simulation.

    Attributes
    ----------
    frequency_cm1 : list[str]
        List of vibrational frequencies in inverse centimeters (cm⁻¹).
        Each entry is a string representation of the frequency value.
    intensity : list[str]
        List of vibrational intensities.
        Each entry is a string representation of the intensity value.
    """

    frequency_cm1: list[str] = Field(
        ...,
        description="List of vibrational frequencies in cm-1.",
    )

    intensity: list[str] = Field(
        ...,
        description="List of intensities in D/Å^2 amu^-1.",
    )

    plot: Optional[str] = None   # base64 PNG image


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
        IRSpectrum,
        AtomsData,
    ] = Field(
        description=(
            "Structured answer to the user's query. Use:\n"
            "1. `str` for general or explanatory responses or SMILES string.\n"
            "2. `VibrationalFrequency` for vibrational frequencies.\n"
            "3. `IRSpectrum` for vibrational frequencies and intensities.\n"
            "4. `ScalarResult` for single numerical properties (e.g. enthalpy).\n"
            "5. `AtomsData` for atomic geometries (XYZ coordinate, etc.) and optimized structures."
        )
    )
