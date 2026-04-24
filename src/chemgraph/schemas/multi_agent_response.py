from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator
from chemgraph.schemas.atomsdata import AtomsData


class WorkerTask(BaseModel):
    """
    Represents a task assigned to an executor agent for performing tool-based computations.

    Attributes:
        task_index (int): The index or ID of the task, typically used to track execution order.
        prompt (str): A natural language prompt that describes the task or request for which
                       the executor is expected to generate tool calls.
        retry_count (int): How many times this task has been previously attempted.
                           Defaults to 0 for new tasks.  When the planner re-dispatches
                           a failed task, the router increments this value automatically.
    """

    task_index: int = Field(..., description="Task index")
    prompt: str = Field(..., description="Prompt to send to executor for tool calls")
    retry_count: int = Field(
        default=0,
        description="Number of previous attempts for this task (0 = first attempt)",
    )


class PlannerResponse(BaseModel):
    """
    Response model from the Planner agent.

    The planner acts as a router: it decides whether to dispatch tasks
    to executor subgraphs (``executor_subgraph``) or to finish
    (``FINISH``) when all work is done.

    Attributes:
        thought_process (str): The planner's reasoning for the current decision.
        next_step (str): The next node to activate — either ``"executor_subgraph"``
            to fan-out tasks or ``"FINISH"`` to end the workflow.
        tasks (list[WorkerTask] | None): Tasks to assign when routing to executors.
    """

    thought_process: str = Field(
        description="Your reasoning for the current decision."
    )
    next_step: Literal["executor_subgraph", "FINISH"] = Field(
        description="The next node to activate in the workflow."
    )
    tasks: list[WorkerTask] = Field(
        default=None,
        description="List of tasks to assign to executor subgraphs.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_planner_payload(cls, data: Any) -> Any:
        """Accept common planner variants and coerce into PlannerResponse shape."""
        if isinstance(data, list):
            return {
                "thought_process": "Delegating parsed tasks to executors.",
                "next_step": "executor_subgraph",
                "tasks": data,
            }

        if isinstance(data, dict):
            normalized = dict(data)
            # Accept legacy "worker_tasks" key
            if "tasks" not in normalized and "worker_tasks" in normalized:
                normalized["tasks"] = normalized.pop("worker_tasks")
            if "tasks" in normalized and "next_step" not in normalized:
                normalized["next_step"] = "executor_subgraph"
            if "tasks" in normalized and "thought_process" not in normalized:
                normalized["thought_process"] = (
                    "Delegating parsed tasks to executors."
                )
            return normalized

        return data


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
    plot : Optional[str]
        Base64-encoded PNG image of the IR spectrum plot.
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
    """Defined structured response to the user."""

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
            "3. `ScalarResult` for single numerical properties (e.g. enthalpy).\n"
            "4. `AtomsData` for atomic geometries (XYZ coordinate, etc.) and optimized structures."
            "5. `InfraredSpectrum` for calculating infrared spectra."
        )
    )
