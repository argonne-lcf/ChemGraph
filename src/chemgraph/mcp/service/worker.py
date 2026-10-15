"""One calculation per process, with no database or bearer credentials."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from pydantic_core import to_jsonable_python

from chemgraph.runtime.context import WorkspaceScope, workspace_scope
from chemgraph.runtime.policy import ExecutionPolicy, PreparedAction, validate


def chemistry_tools():
    """Use the existing validated wrappers; keep their imports out of startup."""
    from chemgraph.runtime.policy import PREPARABLE_TOOLS
    from chemgraph.tools import ase_tools, cheminformatics_tools

    return {name: getattr(ase_tools, name, None) or getattr(cheminformatics_tools, name)
            for name in PREPARABLE_TOOLS}


def policy_dict(policy):
    return to_jsonable_python(asdict(policy))


def execute(action, scope):
    """Explicitly bind and reset inputs and outputs, also usable through to_thread."""
    outputs = {item["path"] for item in action.inputs} if action.tool == "run_ase" else set()
    with workspace_scope(scope, inputs=[item["path"] for item in action.inputs], outputs=outputs):
        try:
            value = chemistry_tools()[action.tool].invoke(action.arguments)
            result = to_jsonable_python(value)
            # Some core tools return a structured failure instead of raising.
            error = isinstance(result, dict) and result.get("status") in {"failure", "error", "failed"}
        except Exception as exc:
            result, error = {"error": str(exc), "error_type": type(exc).__name__}, True
    return {"result": result, "error": error, "registered": sorted(outputs)}


def public_value(value, workspace):
    """Keep server paths out of published results and tool responses."""
    if isinstance(value, str):
        return value.replace(str(workspace) + "/", "").replace(str(workspace), ".")
    if isinstance(value, dict):
        return {key: public_value(item, workspace) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item, workspace) for item in value]
    return value


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(report, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    request_path = Path(sys.argv[1])
    payload = json.loads(request_path.read_text())
    root = request_path.parent.resolve()
    # Limit individual output/log files as well as parent-side total accounting.
    if os.name == "posix":
        import resource

        limit = payload["max_file_bytes"]
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    action = PreparedAction.from_dict(payload["prepared"])
    scope = WorkspaceScope(payload["owner"], root)
    context = SimpleNamespace(owner=scope.owner, workspace=root, scope=scope,
                              thread_id=payload["thread_id"], task_id=payload["task_id"],
                              policy=ExecutionPolicy(**payload["policy"]))
    try:
        validate(context, payload["call"], action)
    except Exception as exc:
        report = {"result": {"error": str(exc)}, "error": True, "registered": []}
    else:
        report = execute(action, scope)
        output = action.arguments.get("params", {}).get("output_results_file")
        if output in report["registered"] and (root / output).suffix == ".json":
            result_file = root / output
            value = json.loads(result_file.read_text())
            write_report(result_file, public_value(value, root))
    write_report(root / ".service-return.json", report)


if __name__ == "__main__":
    main()
