"""Pluggable execution backends for ChemGraph HPC workloads.

This package provides a backend-agnostic interface for submitting
computational tasks to different workflow managers (Parsl,
EnsembleLauncher, Globus Compute, local process pool).

Quick start
-----------
>>> from chemgraph.execution import get_backend, TaskSpec
>>> backend = get_backend()               # reads config.toml / env vars
>>> future = backend.submit(TaskSpec(
...     task_id="test-1",
...     task_type="python",
...     callable=my_function,
...     kwargs={"param": 42},
... ))
>>> result = future.result()
>>> backend.shutdown()

See Also
--------
:mod:`chemgraph.execution.base` -- abstract classes
:mod:`chemgraph.execution.config` -- factory function
"""

from chemgraph.execution.base import ExecutionBackend, TaskSpec
from chemgraph.execution.config import get_backend

__all__ = [
    "ExecutionBackend",
    "TaskSpec",
    "get_backend",
]
