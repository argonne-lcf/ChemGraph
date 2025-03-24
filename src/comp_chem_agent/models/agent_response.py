from pydantic import BaseModel, Field
from typing import Union
from comp_chem_agent.models.atomsdata import AtomsData


class VibrationalFrequency(BaseModel):
    frequency_cm1: list[str] = Field(
        ...,
        description="List of vibrational frequencies in cm-1.",
    )


class ScalarResult(BaseModel):
    value: float = Field(..., description="Scalar numerical result like enthalpy")
    property: str = Field(
        ..., description="Name of the property, e.g. 'enthalpy', 'Gibbs free energy'"
    )
    unit: str = Field(..., description="Unit of the result, e.g. 'eV'")


class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""

    answer: Union[str, ScalarResult, VibrationalFrequency, AtomsData] = Field(
        description=(
            "Structured answer to the user's query. Use:\n"
            "- `str` for general or explanatory responses or SMILES string.\n"
            "- `VibrationalFrequency` for vibrational frequecies\n"
            "- `ScalarResult` for single numerical properties (e.g. enthalpy)\n"
            "- `AtomsData` for atomic geometries and molecular structures"
        )
    )
