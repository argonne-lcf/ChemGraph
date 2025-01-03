from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json

class XTBSimulationInput(BaseModel):
    acc: float = Field(default=1, description="Accuracy for SCC calculation, lower is better.", alias="-a")
    chrg: int = Field(default=0, description="Charge of the molecule.", alias="-c")
    gfn: int = Field(default=2, description="Specify parametrization of GFN-xTB.")
    opt: str = Field(default="normal", description="Perform geometry optimization with specified level", alias="-o")