"""Per-agent native tools. No network access occurs during registration."""

from functools import wraps
from typing import Literal

from langchain_core.tools import tool

from chemgraph.execution.globus_transfer import TransferAuthenticationRequired
from chemgraph.tools.alcf_iri_core import IRIAuthenticationRequired, IRIRequestError
from chemgraph.tools.hpc.models import BatchRequest
from chemgraph.tools.hpc.service import HPCService


def _errors(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except IRIAuthenticationRequired as exc:
            return {"error": "authentication_required", "message": str(exc)}
        except TransferAuthenticationRequired as exc:
            return {
                "error": "authentication_required",
                "message": f"{exc} Run the login command in a terminal; if staging started, retry with a fresh run directory.",
            }
        except IRIRequestError as exc:
            return {
                "error": "iri_request_failed",
                "http_status": exc.status_code,
                "message": str(exc),
            }
        except (ValueError, KeyError, OSError) as exc:
            return {"error": "invalid_run", "message": str(exc)}
        except Exception as exc:
            # SDK/transport exceptions may contain headers or response bodies.
            return {
                "error": "operation_failed",
                "type": type(exc).__name__,
                "message": "Inspect saved evidence; do not automatically repeat mutations. Verify authentication and collection consent.",
            }

    return call


def create_hpc_tools(config, *, service=None):
    service = service or HPCService(config)

    @tool
    @_errors
    def hpc_list_targets() -> dict:
        """List configured HPC targets, path mappings, and resource defaults."""
        return service.list_targets()

    @tool
    @_errors
    def hpc_transfer_files(
        run_dir: str,
        files: list[str],
        direction: Literal["stage", "retrieve"],
        target: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Stage selected run-relative files once, or retrieve remote files into run_dir/retrieved.

        Staging freezes inputs and assigns remote_directory. Never edits scripts or
        input JSON: use relative compute paths. Changed inputs require a fresh run.
        Retrieval overwrite must be explicitly requested. Returns a transfer ID.
        """
        if direction == "stage":
            if not target:
                raise ValueError("Staging requires a configured target.")
            if overwrite:
                raise ValueError("Staged inputs cannot be overwritten.")
            return service.stage(run_dir, target, files)
        return service.retrieve(run_dir, files, overwrite=overwrite)

    @tool
    @_errors
    def hpc_transfer_status(run_dir: str, transfer_id: str | None = None) -> dict:
        """Inspect a recorded Globus transfer without submitting computation."""
        return service.transfer_status(run_dir, transfer_id)

    @tool
    @_errors
    def hpc_submit_job(run_dir: str, request: BatchRequest) -> dict:
        """Submit one reviewed PBS batch after staging succeeds; never transfers files.

        Resources are IRI JobSpec settings; #PBS comments are ignored. Repeated
        unchanged accepted requests return the existing job ID. Unknown acceptance
        must be reconciled with hpc_job_status, never retried in a new run blindly.
        """
        return service.submit(run_dir, request)

    @tool
    @_errors
    def hpc_job_status(run_dir: str) -> dict:
        """Inspect saved job evidence, reconcile uncertainty, and query scheduler history.

        Scheduler state does not establish scientific success; inspect result files.
        """
        return service.status(run_dir)

    @tool
    @_errors
    def hpc_list_jobs(
        target: str,
        historical: bool = False,
        limit: int = 100,
        offset: int = 0,
        filters: dict | None = None,
    ) -> dict:
        """List scheduler jobs with pagination; filters: states, owner, jobIds, queue, accountingId."""
        return {
            "jobs": service.list_jobs(
                target,
                historical=historical,
                limit=limit,
                offset=offset,
                filters=filters,
            )
        }

    @tool
    @_errors
    def hpc_cancel_job(run_dir: str, job_id: str) -> dict:
        """Request cancellation of this run's exact full job ID; monitor afterward."""
        return service.cancel(run_dir, job_id)

    @tool
    @_errors
    def hpc_read_file(
        run_dir: str,
        path: str,
        offset: int = 0,
        size: int = 16384,
        operation_id: str | None = None,
    ) -> dict:
        """Read bounded remote text/JSON. Poll a returned task_id using operation_id.

        Paths are relative to this run. Size is at most 65536 bytes. Use Globus
        retrieval for artifacts or truncated responses. IRI filesystem access
        may require additional facility permissions.
        """
        return service.inspect(
            run_dir, path, offset=offset, size=size, operation_id=operation_id
        )

    @tool
    @_errors
    def hpc_list_files(
        run_dir: str, path: str = ".", operation_id: str | None = None
    ) -> dict:
        """List a remote run directory with a bounded response; poll returned task_id with operation_id."""
        return service.inspect(run_dir, path, operation="ls", operation_id=operation_id)

    return [
        hpc_list_targets,
        hpc_transfer_files,
        hpc_transfer_status,
        hpc_submit_job,
        hpc_job_status,
        hpc_list_jobs,
        hpc_cancel_job,
        hpc_read_file,
        hpc_list_files,
    ]


def create_hpc_registry(config, *, names=None, service=None, human_supervised=False):
    """Bind HPC tools, then apply restrictions (including an empty selection)."""
    from chemgraph.registry.tools import ToolRegistry

    registry = ToolRegistry()
    for entry in create_hpc_tools(config, service=service):
        registry.register(entry, tags=("hpc", "pbs"))
    if names is None:
        names = [
            spec.name
            for spec in registry.specs()
            if human_supervised or not spec.interactive
        ]
    return registry.select(names)
