"""Configuration and validated plans for the gRASPA MCP graph."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemgraph.schemas.graspa_schema import SimulationCondition


class GraspaWorkflowOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_directory: str | None = Field(default=None, min_length=1)
    poll_interval_seconds: float = Field(default=15, gt=0, allow_inf_nan=False)
    wait_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    resume: bool = False

    @model_validator(mode="after")
    def resume_directory(self):
        if self.resume and not self.run_directory:
            raise ValueError("Resuming requires an explicit run_directory")
        return self


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


class GraspaTask(BaseModel):
    task_index: int = Field(gt=0)
    prompt: str = Field(min_length=1)


class GraspaPlan(BaseModel):
    """Logical ensemble tasks, not one LLM task per CIF."""

    tasks: list[GraspaTask] = Field(min_length=1)
    analysis: GraspaAnalysis | None = None

    @model_validator(mode="after")
    def unique_tasks(self):
        if len({task.task_index for task in self.tasks}) != len(self.tasks):
            raise ValueError("Plan task indices must be unique")
        return self
