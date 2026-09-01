"""Backend-aware MCP server for calculator-selectable MLIP calculations.

The public request contains one scientific calculation configuration. The
``calculator.backend`` field selects ASE, NVIDIA ALCHEMI, or Rootstock, while
``[execution] backend`` in ``config.toml`` independently selects where this
Python task runs (local, Parsl, Globus Compute, or EnsembleLauncher).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from chemgraph.execution.base import TaskSpec
from chemgraph.execution.config import get_transfer_manager
from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.mcp.transfer_tools import register_transfer_tools
from chemgraph.schemas.mlip_input import MLIPBatchInputSchema, MLIPInputSchema
from chemgraph.tools.ase_core import _resolve_existing_path, _resolve_path
from chemgraph.tools.run_mlip_core import run_mlip_batch_core, run_mlip_core

logger = logging.getLogger(__name__)

_JOBS_FILE = Path("~/.chemgraph/mlip_jobs.json").expanduser()

mcp = CGFastMCP(
    name="ChemGraph MLIP Tools",
    instructions=(
        "Run MLIP calculations using one request configuration. "
        "calculator.backend selects ASE, NVIDIA ALCHEMI, or Rootstock; the "
        "server's execution configuration independently selects local, Parsl, "
        "Globus Compute, or EnsembleLauncher execution. A batch is submitted "
        "as one backend task so its calculator or model can be reused. For "
        "non-shared remote filesystems, pre-stage structures and local "
        "checkpoint files, then provide remote absolute input and output paths."
    ),
)


def run_mlip(params: MLIPInputSchema) -> dict:
    """Run one MLIP energy calculation or fixed-cell geometry optimization."""
    return run_mlip_core(params)


def run_mlip_batch(params: MLIPBatchInputSchema) -> dict:
    """Run an ordered MLIP batch as one execution-backend task."""
    return run_mlip_batch_core(params)


def _backend_shares_fs() -> bool:
    """Return whether the configured execution worker sees server-local files."""
    backend = getattr(mcp, "_backend", None)
    return getattr(backend, "shares_filesystem", True)


def _resolved_local_structure(path: str) -> str | None:
    """Return a server-local structure path, or ``None`` for a remote path."""
    resolved = _resolve_existing_path(path)
    return resolved if os.path.isfile(resolved) else None


def _resolved_local_directory(path: str) -> str | None:
    """Return a server-local input directory, including CHEMGRAPH_LOG_DIR."""
    candidate = Path(path)
    if candidate.is_dir():
        return str(candidate.resolve())
    resolved = Path(_resolve_path(path))
    return str(resolved.resolve()) if resolved.is_dir() else None


def _resolved_local_optional_file(path: str | None) -> str | None:
    """Resolve an optional checkpoint-like value only when it is a local file."""
    if not path:
        return None
    resolved = _resolve_existing_path(path)
    return resolved if os.path.isfile(resolved) else None


def _local_request_files(
    params: MLIPInputSchema | MLIPBatchInputSchema,
) -> list[str]:
    """Collect files that would be inaccessible on a non-shared worker."""
    local: list[str] = []
    if isinstance(params, MLIPInputSchema):
        structure = _resolved_local_structure(params.input_structure_file)
        if structure is not None:
            local.append(structure)
    else:
        for path in params.input_structure_files or []:
            structure = _resolved_local_structure(path)
            if structure is not None:
                local.append(structure)
        if params.input_structure_directory is not None:
            directory = _resolved_local_directory(params.input_structure_directory)
            if directory is not None:
                local.append(directory)

    checkpoint = _resolved_local_optional_file(params.model.checkpoint)
    if checkpoint is not None:
        local.append(checkpoint)
    weights = getattr(params.calculator, "weights", None)
    weights_file = _resolved_local_optional_file(weights)
    if weights_file is not None:
        local.append(weights_file)
    return local


def _validate_remote_paths(
    params: MLIPInputSchema | MLIPBatchInputSchema,
) -> None:
    """Require explicit, pre-staged paths for non-shared execution backends."""
    local_files = _local_request_files(params)
    if local_files:
        listed = ", ".join(local_files)
        raise ValueError(
            "The execution backend does not share the MCP server filesystem. "
            "Pre-stage these inputs (for example with transfer_files) and pass "
            f"their remote paths instead: {listed}"
        )

    if isinstance(params, MLIPInputSchema):
        output_path = params.output_results_file
        field_name = "output_results_file"
    else:
        output_path = params.output_results_directory
        field_name = "output_results_directory"
    if not os.path.isabs(output_path):
        raise ValueError(
            f"{field_name} must be an absolute path on a non-shared execution "
            "backend."
        )


def _needs_execution_gpu(
    params: MLIPInputSchema | MLIPBatchInputSchema,
) -> bool:
    """Return whether the outer execution task directly evaluates on CUDA."""
    calculator = params.calculator
    device = getattr(calculator, "device", None)
    return calculator.backend != "rootstock" and bool(
        device and device.lower().startswith("cuda")
    )


def _mlip_transport_hook(task: TaskSpec) -> TaskSpec:
    """Validate remote paths and add a GPU resource hint before submission."""
    if task.callable not in (run_mlip, run_mlip_batch):
        return task

    raw_params: Any = task.kwargs.get("params")
    schema = (
        MLIPInputSchema if task.callable is run_mlip else MLIPBatchInputSchema
    )
    params = schema.model_validate(raw_params)
    if not _backend_shares_fs():
        _validate_remote_paths(params)
    if _needs_execution_gpu(params):
        task.gpus_per_task = max(task.gpus_per_task, 1)
    task.kwargs = {"params": params.model_dump(mode="json")}
    return task


mcp.set_pre_submit_hook(_mlip_transport_hook)
mcp.tool(name="run_mlip")(run_mlip)
mcp.tool(name="run_mlip_batch")(run_mlip_batch)
mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})

_transfer_manager = get_transfer_manager()
if _transfer_manager is not None:
    register_transfer_tools(mcp, _transfer_manager)
    logger.info("Registered Globus Transfer tools on MLIP MCP server.")


if __name__ == "__main__":
    try:
        run_mcp_server(mcp, default_port=9006)
    finally:
        mcp.shutdown_backend()
