#!/usr/bin/env python3
"""Run a small live Globus/IRI/PBS demonstration from this checkout."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import tomllib


SOURCE = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SOURCE))
INPUT = b"1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"
OUTPUTS = ["result.txt", "stdout.txt", "stderr.txt"]


class DemoError(RuntimeError):
    """A diagnostic safe to print without API headers or token contents."""


def report(step, **details):
    print(json.dumps({"step": step, **details}), flush=True)


def invoke(registry, name, **arguments):
    result = registry.get(name).invoke(arguments)
    if "error" in result:
        raise DemoError(json.dumps({"tool": name, **result}))
    return result


def load_config(path, target_name):
    from chemgraph.tools.hpc import HPCConfig

    with path.open("rb") as stream:
        data = tomllib.load(stream)["hpc"]
    if "YOUR_" in json.dumps(data):
        raise DemoError("Replace every YOUR_* placeholder in your local config first.")
    config = HPCConfig.model_validate(data)
    if target_name not in config.targets:
        raise DemoError(f"Unknown target {target_name!r}; choose from {sorted(config.targets)}.")
    target = config.targets[target_name]
    if target.resources.node_count != 1 or target.resources.duration > 300:
        raise DemoError("This demo uses one node and at most 300 seconds of walltime.")
    return config


def preflight(service, target_name):
    from chemgraph.execution.globus_transfer import transfer_error_details

    target = service.config.targets[target_name]
    if not Path(target.local_root).is_dir():
        raise DemoError(f"Create your configured local_root first: {target.local_root}")
    client = service.manager(target)._get_transfer_client()
    for side in ("local", "remote"):
        collection = getattr(target, side + "_collection")
        path = getattr(target, side + "_collection_root")
        try:
            client.operation_ls(collection, path=path, limit=1)
        except Exception as exc:
            details = transfer_error_details(exc)
            hint = "Verify collection permissions and terminal login."
            if "GCDisconnected" in details.get("code", ""):
                hint = "Start Globus Connect Personal on the collection host and wait for it to connect."
            raise DemoError(json.dumps({"side": side, "collection": collection,
                                       "path": path, "hint": hint, **details})) from None
        report("collection accessible", side=side, collection=collection, path=path)
    compute = service.iri.resolve_resource(target.compute_resource)
    storage = service.iri.resolve_resource(target.storage_resource)
    service.iri.jobs(compute, limit=1, filters={"accountingId": target.project})
    report("preflight passed", compute=compute, storage=storage,
           project=target.project, queue=target.queue, resources=target.resources.model_dump())


def poll(read, done, timeout, interval, label):
    deadline = time.monotonic() + timeout
    while True:
        result = read()
        report(label, **result)
        if done(result):
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DemoError(f"Timed out waiting for {label}; preserve this run and inspect its status. "
                            f"No resubmission occurred. Last observation: {json.dumps(result)}")
        time.sleep(min(interval, remaining))


def wait_transfer(registry, root, task_id, timeout, interval):
    def done(result):
        if result.get("status") == "FAILED":
            raise DemoError(f"Transfer {task_id} failed. Preserve the run and inspect Globus task events.")
        return result.get("status") == "SUCCEEDED"

    poll(lambda: invoke(registry, "hpc_transfer_status", run_dir=str(root), transfer_id=task_id),
         done, timeout, interval, "transfer status")


def wait_job(registry, root, timeout, interval):
    def done(result):
        if result.get("state") in {"not_submitted", "prepared", "rejected"}:
            raise DemoError("No accepted job is recorded. Inspect submission.json; this demo never resubmits.")
        scheduler = result.get("scheduler", {})
        if not isinstance(scheduler, dict):
            raise DemoError(f"Unexpected scheduler status for {root}: {scheduler!r}")
        state = str(scheduler.get("state", "")).lower()
        if state in {"failed", "canceled", "cancelled"}:
            raise DemoError(f"Job ended with state {state}. Inspect saved job and output evidence.")
        if state == "completed" and scheduler.get("exit_code") not in (None, 0):
            raise DemoError(f"Job failed with exit code {scheduler['exit_code']}.")
        return state == "completed"

    return poll(lambda: invoke(registry, "hpc_job_status", run_dir=str(root)),
                done, timeout, interval, "job status")


def collect(registry, root, timeout, interval):
    status = wait_job(registry, root, timeout, interval)
    history_path = root / ".hpc-retrievals.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        if len(history) != 1 or history[0].get("files") != OUTPUTS or not history[0].get("transfer_id"):
            raise DemoError("Inspect the existing retrieval evidence manually; no new transfer was requested.")
        transfer = history[0]
    else:
        transfer = invoke(registry, "hpc_transfer_files", run_dir=str(root),
                          direction="retrieve", files=OUTPUTS)
    report("retrieval", transfer_id=transfer["transfer_id"])
    wait_transfer(registry, root, transfer["transfer_id"], timeout, interval)
    path = root / "retrieved/result.txt"
    try:
        result = dict(line.split("=", 1) for line in path.read_text().splitlines())
    except (OSError, ValueError) as exc:
        raise DemoError(f"Missing or malformed demo result at {path} ({type(exc).__name__}).") from None
    expected = {"sum": "55", "input_sha256": hashlib.sha256(INPUT).hexdigest(),
                "pbs_job_id": status.get("job_id")}
    failures = {key: {"expected": value, "observed": result.get(key)}
                for key, value in expected.items() if result.get(key) != value}
    if not result.get("hostname"):
        failures["hostname"] = {"expected": "nonempty", "observed": result.get("hostname")}
    if failures:
        raise DemoError(json.dumps({"run_dir": str(root), "output_checks_failed": failures}))
    report("verified", **result, run_dir=str(root), outputs=str(root / "retrieved"))


def show_status(registry, root):
    report("staging", **invoke(registry, "hpc_transfer_status", run_dir=str(root)))
    report("job", **invoke(registry, "hpc_job_status", run_dir=str(root)))
    history_path = root / ".hpc-retrievals.json"
    if history_path.exists():
        for record in json.loads(history_path.read_text()):
            task_id = record.get("transfer_id")
            if task_id:
                report("retrieval status", **invoke(registry, "hpc_transfer_status",
                                                    run_dir=str(root), transfer_id=task_id))
            else:
                report("retrieval status", state=record.get("state", "unknown"),
                       submission_id=record.get("submission_id"), transfer_id=None)


def run(config, registry, target_name, root, timeout, interval):
    target = config.targets[target_name]
    boundary = Path(target.local_root).resolve()
    if not root.is_absolute() or root.resolve() == boundary or not root.resolve().is_relative_to(boundary):
        raise DemoError("Use an absolute, new run directory beneath the target local_root.")
    root.mkdir()  # Refuse to restart a demonstration in any existing directory.
    report("run directory", run_dir=str(root))
    (root / "inputs").mkdir()
    (root / "inputs/numbers.txt").write_bytes(INPUT)
    shutil.copyfile(Path(__file__).with_name("launch.sh"), root / "launch.sh")
    staged = invoke(registry, "hpc_transfer_files", run_dir=str(root), target=target_name,
                    direction="stage", files=["launch.sh", "inputs/numbers.txt"])
    report("staged", transfer_id=staged["transfer_id"], remote_directory=staged["remote_directory"])
    wait_transfer(registry, root, staged["transfer_id"], timeout, interval)
    request = {"target": target_name, "launch_script": "launch.sh", "project": target.project,
               "queue": target.queue, "resources": target.resources.model_dump(), "expected_outputs": OUTPUTS}
    report("submitting once", request=request)
    result = invoke(registry, "hpc_submit_job", run_dir=str(root), request=request)
    report("submission", **result)
    collect(registry, root, timeout, interval)


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Use a finite positive number of seconds.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "run", "status", "collect"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", default="polaris")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--timeout", type=positive_seconds, default=600,
                        help="Deadline per transfer/job wait, including queue time (seconds).")
    parser.add_argument("--poll-interval", type=positive_seconds, default=5)
    args = parser.parse_args(argv)
    if args.mode != "check" and args.run_dir is None:
        parser.error("--run-dir is required for run, status and collect")
    transfer_error_details = None
    transfer_errors = ()
    safe_errors = ()
    try:
        expected_package = SOURCE / "chemgraph/__init__.py"
        if not expected_package.is_file():
            raise DemoError(f"This checkout is incomplete: {expected_package} is missing. "
                            "Restore the checkout before retrying; preserve your config and run directory.")
        import chemgraph

        package = getattr(chemgraph, "__file__", None)
        if package is None or Path(package).resolve() != expected_package:
            raise DemoError("Another ChemGraph checkout is loaded; start this script in a new Python process.")
        from globus_sdk import TransferAPIError
        from chemgraph.execution.globus_transfer import TransferAuthenticationRequired, transfer_error_details
        from chemgraph.tools.alcf_iri_core import IRIAuthenticationRequired, IRIRequestError
        from chemgraph.tools.hpc.service import HPCService
        from chemgraph.tools.hpc.tools import create_hpc_registry

        safe_errors = (IRIAuthenticationRequired, IRIRequestError, TransferAuthenticationRequired)
        transfer_errors = (TransferAPIError,)
        config = load_config(args.config, args.target)
        service = HPCService(config)
        registry = create_hpc_registry(config, service=service)
        report("runtime", python=sys.executable, chemgraph=str(package), config=str(args.config))
        if args.mode in {"check", "run"}:
            preflight(service, args.target)
        if args.mode == "run":
            run(config, registry, args.target, args.run_dir, args.timeout, args.poll_interval)
        elif args.mode == "status":
            show_status(registry, args.run_dir)
        elif args.mode == "collect":
            collect(registry, args.run_dir, args.timeout, args.poll_interval)
        return 0
    except DemoError as exc:
        report("stopped", message=str(exc))
    except ImportError as exc:
        report("stopped", type=type(exc).__name__,
               message=f"Cannot load the demo runtime from {SOURCE}. "
                       "Restore missing checkout files and install this checkout's required dependencies.")
    except Exception as exc:
        message = str(exc) if isinstance(exc, safe_errors) else "Check configuration, login and saved evidence. No automatic retry or cleanup was performed."
        details = transfer_error_details(exc) if isinstance(exc, transfer_errors) else {}
        report("stopped", type=type(exc).__name__, message=message, **details)
    return 1


if __name__ == "__main__":
    sys.exit(main())
