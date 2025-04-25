from pydantic import BaseModel, Field
from typing import Union
from comp_chem_agent.models.atomsdata import AtomsData


class WorkerTask(BaseModel):
    task_index: int = Field(..., description="Task index")
    prompt: str = Field(..., description="Prompt to send to worker for tool calls")

class TaskDecomposerResponse(BaseModel):
    worker_tasks: list[WorkerTask]= Field(..., description="List of task to assign for Worker")

