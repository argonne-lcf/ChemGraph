from pydantic import BaseModel, Field
from typing import Union
from chemgraph.models.atomsdata import AtomsData


class WorkerTask(BaseModel):
    """
    Represents a task assigned to a worker agent for performing tool-based computations.

    Attributes:
        task_index (int): The index or ID of the task, typically used to track execution order.
        prompt (str): A natural language prompt that describes the task or request for which
                      the worker is expected to generate tool calls.
    """

    task_index: int = Field(..., description="Task index")
    prompt: str = Field(..., description="Prompt to send to worker for tool calls")


class PlannerResponse(BaseModel):
    """
    Response model from the Task Decomposer agent containing a list of tasks.

    Attributes:
        worker_tasks (list[WorkerTask]): A list of tasks that are to be assigned
        to Worker agents for tool execution or computation.
    """

    worker_tasks: list[WorkerTask] = Field(
        ..., description="List of task to assign for Worker"
    )


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
    """Defined structured response to the user."""

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
            "- `VibrationalFrequency` for vibrational frequencies.\n"
            "- `ScalarResult` for single numerical properties (e.g. enthalpy).\n"
            "- `AtomsData` for atomic geometries and molecular structures."
        )
    )
