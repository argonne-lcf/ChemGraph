from pydantic import BaseModel, Field
from typing import Union, Optional
from chemgraph.schemas.atomsdata import AtomsData


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


class InfraredSpectrum(BaseModel):
    """
    Schema for calculating infrared spectrum from a simulation.

    Attributes
    ----------
    frequency_spec_cm1 : list[str]
        List of range of frequencies in inverse centimeters (cm⁻¹)
        Each entry is a string representation of the frequency value.
    intensity_spec_D2A2amu1 : list[str]
        List of range of intensities in (D/Å)^2 amu⁻¹
        Each entry is a string representation of the intensity value.
    """
    frequency_spec_cm1: list[str] = Field(
        ...,
        description="Range of frequencies for plotting spectrum in cm-1.",
    )
    
    intensity_spec_D2A2amu1: list[str] = Field(
        ...,
        description="Values of intensities for plotting spectrum in (D/Å)^2 amu^-1.",
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

    text_answer: Optional[str] = Field(
        default=None,
        description="General or explanatory responses or SMILES string."
    )
    scalar_answer: Optional[ScalarResult] = Field(
        default=None,
        description="Single numerical properties (e.g. enthalpy)."
    )
    vibrational_answer: Optional[VibrationalFrequency] = Field(
        default=None,
        description="Vibrational frequencies."
    )
    ir_spectrum: Optional[IRSpectrum] = Field(
        default=None,
        description="Infrared spectra."
    )
    atoms_data: Optional[AtomsData] = Field(
        default=None,
        description="Atomic geometries (XYZ coordinate, etc.) and optimized structures."
    )
