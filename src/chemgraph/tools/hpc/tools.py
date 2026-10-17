"""Per-agent native tools. No network access occurs during registration."""

from functools import wraps
from typing import Literal

from langchain_core.tools import tool

from chemgraph.execution.globus_transfer import TransferAuthenticationRequired
from chemgraph.tools.alcf_iri_core import IRIAuthenticationRequired, IRIRequestError
from chemgraph.tools.hpc.models import MANAGED_HPC_TOOL, BatchRequest, HPCConfig
from chemgraph.tools.hpc.service import HPCService, TransferOperationError


def _errors(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TransferOperationError as exc:
            return exc.details
        except IRIAuthenticationRequired as exc:
            return {"error": "authentication_required", "message": str(exc)}
        except TransferAuthenticationRequired as exc:
            return {
                "error": "authentication_required",
                "message": f"{exc} Run the login command in a terminal; preserve all recorded transfer attempts.",
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
        An unknown outcome is resolved by repeating the same call, which replays
        the saved submission ID. Retry a confirmed rejection, or a staging
        transfer that reached FAILED, after correcting it.
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
    def hpc_submit_job(run_dir: str, request: BatchRequest, resubmit: bool = False) -> dict:
        """Submit one reviewed PBS batch after staging succeeds; never transfers files.

        Set project, queue and resources (node_count, duration in seconds,
        filesystems) explicitly so the review shows what will run; take
        defaults from hpc_list_targets. Resources are IRI JobSpec settings;
        #PBS comments are ignored. Repeated unchanged accepted requests return
        the existing job ID. Set resubmit=true only after the scheduler
        confirmed a rejection (state=rejected). Unknown acceptance must be
        reconciled with hpc_job_status, never resubmitted.
        """
        request = BatchRequest.model_validate(request)
        explicit = request.resources.model_fields_set if request.resources else set()
        missing = [name for name in ("project", "queue") if getattr(request, name) is None]
        missing += [f"resources.{name}" for name in EXPLICIT_RESOURCES if name not in explicit]
        if missing:
            # The reviewed arguments must contain every consequential setting.
            return {
                "error": "explicit_settings_required",
                "missing": missing,
                "configured_defaults": service.effective_settings(request.target, request),
                "message": "Nothing was submitted. Repeat the call with these settings explicit "
                "in the request so the review shows the account, queue and resources.",
            }
        return service.submit(run_dir, request, resubmit=resubmit)

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
        """Request cancellation of this run's exact full job ID; monitor afterward.

        Accepts the accepted job or a duplicate_jobs entry from hpc_job_status,
        which scheduler history must still show belongs to this run.
        """
        return service.cancel(run_dir, job_id)

    @tool
    @_errors
    def hpc_read_file(
        run_dir: str,
        path: str,
        offset: int = 0,
        size: int = 16384,
    ) -> dict:
        """Read bounded remote text/JSON. Poll a returned task_id with hpc_operation_status.

        Paths are relative to this run. Size is at most 65536 bytes. Use Globus
        retrieval for artifacts or truncated responses. IRI filesystem access
        may require additional facility permissions.
        """
        return service.inspect(run_dir, path, offset=offset, size=size)

    @tool
    @_errors
    def hpc_list_files(run_dir: str, path: str = ".") -> dict:
        """List a remote run directory with a bounded response; poll a returned task_id with hpc_operation_status."""
        return service.inspect(run_dir, path, operation="ls")

    @tool
    @_errors
    def hpc_operation_status(run_dir: str, operation_id: str) -> dict:
        """Poll a task_id returned by hpc_read_file or hpc_list_files for this run, with the same bounds."""
        return service.operation_status(run_dir, operation_id)

    tools = [
        hpc_list_targets,
        hpc_transfer_files,
        hpc_transfer_status,
        hpc_submit_job,
        hpc_job_status,
        hpc_list_jobs,
        hpc_cancel_job,
        hpc_read_file,
        hpc_list_files,
        hpc_operation_status,
    ]
    for entry in tools:
        entry.metadata = {**(entry.metadata or {}), MANAGED_HPC_TOOL: True}
    return tools


# Resource fields hpc_submit_job requires in its reviewed arguments.
EXPLICIT_RESOURCES = ("node_count", "duration", "filesystems")


# Transfers, submissions and cancellations change remote state or use an allocation.
HPC_MUTATION_TOOLS = ("hpc_transfer_files", "hpc_submit_job", "hpc_cancel_job")


def create_hpc_registry(
    config, *, base=None, names=None, service=None, human_supervised=False, attached=(),
):
    """Add HPC tools to ``base`` (default: built-ins), then apply restrictions.

    An explicit ``names`` list, including an empty one, selects from the
    combined catalog. Without ``base`` or ``names``, built-ins named by
    ``attached`` tools are left out, as in the default catalog; HPC tools never
    are, so an attached tool reusing an HPC name still conflicts. Mutation
    tools keep their review under approval bypass unless the configuration
    explicitly allows unreviewed mutations.
    """
    from chemgraph.graphs.workspace import REVIEW_REQUIRED_TAG, default_tool_registry
    from chemgraph.registry.tools import ToolRegistry

    config = HPCConfig.model_validate(config)
    if base is not None:
        registry = base.select(base.names())
    elif names is None:
        registry = default_tool_registry(attached, human_supervised=human_supervised)
    else:
        registry = ToolRegistry()
    for entry in create_hpc_tools(config, service=service):
        tags = ("hpc", "pbs")
        if entry.name in HPC_MUTATION_TOOLS and not config.allow_unreviewed_mutations:
            tags += (REVIEW_REQUIRED_TAG,)
        registry.register(entry, tags=tags)
    if names is None:
        names = [
            spec.name
            for spec in registry.specs()
            if base is not None or human_supervised or not spec.interactive
        ]
    return registry.select(names)
