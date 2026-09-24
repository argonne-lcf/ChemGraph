"""Configuration and validated plans for the gRASPA MCP graph."""

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemgraph.schemas.graspa_schema import SimulationCondition, graspa_input_schema_ensemble


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


class GraspaRequestContract(BaseModel):
    """Authoritative workload for one ensemble on a shared filesystem.

    ``request`` uses a compact directory reference. ``sources`` contains the
    resolved selection, including duplicates, and never enters model messages.
    """

    model_config = ConfigDict(extra="forbid")

    request: graspa_input_schema_ensemble
    sources: list[str] = Field(min_length=1)
    analysis: GraspaAnalysis

    @model_validator(mode="after")
    def shared_directory(self):
        if self.request.remote_structure_directory or not isinstance(self.request.input_structures, str):
            raise ValueError("A request contract requires a shared input directory")
        if self.request.output_directory is None:
            raise ValueError("A request contract requires an explicit output_directory")
        if any(not source.strip() for source in self.sources):
            raise ValueError("Contract sources must be nonempty paths")
        for condition in (self.analysis.adsorption, self.analysis.desorption):
            if condition is not None and condition not in self.request.conditions:
                raise ValueError("Contract analysis conditions must be included in the request")
        return self

    def check_plan(self, plan: GraspaPlan) -> None:
        if len(plan.tasks) != 1:
            raise ValueError("Request contract requires exactly one ensemble task")
        if plan.analysis != self.analysis:
            raise ValueError("Plan analysis must match the requested conditions and top_fraction")

    def check_request(self, params: dict, *, frozen: bool = False) -> None:
        expected = self.request.model_dump(mode="json", exclude={"output_result_file", "discovery_timeout_seconds"})
        changed = [key for key, value in expected.items()
                   if (key != "input_structures" or not frozen) and params.get(key) != value]
        if changed:
            raise ValueError("Request contract mismatch: " + ", ".join(changed))
        if frozen and (not isinstance(params.get("input_structures"), list)
                       or Counter(params["input_structures"]) != Counter(self.sources)):
            raise ValueError("Discovered input_structures must match all contract sources, including duplicates")


class GraspaWorkflowOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_directory: str | None = Field(default=None, min_length=1)
    poll_interval_seconds: float = Field(default=15, gt=0, allow_inf_nan=False)
    wait_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    resume: bool = False
    request_contract: GraspaRequestContract | None = None

    @model_validator(mode="after")
    def resume_directory(self):
        if self.resume and not self.run_directory:
            raise ValueError("Resuming requires an explicit run_directory")
        return self
