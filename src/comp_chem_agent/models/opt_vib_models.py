from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json


class RouterResponse(BaseModel):
    next_agent: str = Field(
        description="One of the following: ParameterInputAgent or EndAgent"
    )
    reason: str = Field(description="Explain your choice.")


class ASEFeedbackResponse(BaseModel):
    next_agent: str = Field(
        description="One of the following: ASEParameterAgent or EndAgent"
    )
    feedback: str = Field(description="Feedback to the simulation results.")


class QCEngineFeedbackResponse(BaseModel):
    next_agent: str = Field(
        description="One of the following: QCEngineParameterAgent or EndAgent"
    )
    feedback: str = Field(description="Feedback to the simulation results.")
