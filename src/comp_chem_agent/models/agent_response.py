from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json

class RouterResponse(BaseModel):
    next_agent: str = Field(
        description="One of the following: ParameterInputExpert or EndAgent"
    )
    reason: str = Field(
        description="Explain your choice."
    )






