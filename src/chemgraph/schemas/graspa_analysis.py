"""Validated conditions for gRASPA analysis tools."""

from pydantic import BaseModel, Field, model_validator

from chemgraph.schemas.graspa_schema import SimulationCondition


class GraspaAnalysis(BaseModel):
    """One explicitly requested uptake point or adsorption/desorption pair."""

    adsorption: SimulationCondition
    desorption: SimulationCondition | None = None
    top_fraction: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def distinct_conditions(self):
        if self.desorption == self.adsorption:
            raise ValueError("Adsorption and desorption conditions must differ")
        return self

