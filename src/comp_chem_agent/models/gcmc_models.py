from pydantic import BaseModel, Field
from typing_extensions import TypedDict, List

class StepDescription(BaseModel):
    index: int = Field(
        description="The sequential identifier for this step in the workflow."
    )
    task: str = Field(
        description="A detailed description of the task to be performed, including the relevant software or tools."
    )
    tools: List[str] = Field(
        description="A list of required tools needed to successfully complete this task."
    )
    agent: str = Field(
        description="The designated agent responsible for executing this task. Options: DataQueryAgent, GeoOptAgent, ChargeAgent, GCMCAgent, PostprocessAgent."
    )

class PlannerResponse(BaseModel):
    steps: List[StepDescription] = Field(
        description="An ordered list of steps outlining the tasks necessary to achieve the goal."
    )
