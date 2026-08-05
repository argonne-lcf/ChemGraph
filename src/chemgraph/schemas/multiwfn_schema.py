"""Input schema for scripted Multiwfn analyses."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class MultiwfnInputSchema(BaseModel):
    """Parameters for one non-interactive Multiwfn invocation."""

    input_file: str = Field(
        min_length=1,
        description=(
            "Path to a wavefunction or other input file supported by Multiwfn."
        ),
    )
    menu_inputs: list[str] = Field(
        min_length=1,
        description=(
            "Exact Multiwfn menu responses in order. Use an empty string to press "
            "Enter. Include the menu responses needed to exit Multiwfn cleanly."
        ),
    )
    timeout_s: float = Field(
        default=600.0,
        gt=0,
        allow_inf_nan=False,
        description="Maximum wall time in seconds before terminating Multiwfn.",
    )

    @field_validator("menu_inputs")
    @classmethod
    def validate_menu_inputs(cls, values: list[str]) -> list[str]:
        """Keep one response per item while allowing an empty Enter response."""
        for value in values:
            if "\x00" in value:
                raise ValueError("Multiwfn menu responses cannot contain NUL bytes.")
            if "\n" in value or "\r" in value:
                raise ValueError(
                    "Each Multiwfn menu response must be a single line."
                )
        return values


class MultiwfnResult(TypedDict):
    """Structured metadata returned by a Multiwfn batch invocation."""

    status: Literal["success", "failure", "timeout"]
    return_code: int | None
    duration_s: float
    executable: str
    input_file: str
    run_directory: str
    stdin_file: str
    stdout_file: str
    stderr_file: str
    artifacts: list[str]
    stdout_tail: str
    stderr_tail: str
