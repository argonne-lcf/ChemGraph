"""Write RUN/launch.sh, RUN/request.json, and RUN/calculate.py from ./target.toml.

Run from the workspace that holds target.toml, after writing RUN/input.json and
the structure. An active submission marker or accepted job makes this refuse, so an
attempted request is never edited. After a definite rejection it may rewrite
request.json; once staged, launch.sh and calculate.py must stay unchanged.

Exit codes: 0 files written, 1 error.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tomllib

SKILLS = Path(__file__).resolve().parents[2]
TEMPLATE = SKILLS / "iri-hpc/assets/launch.sh.template"
CALCULATE = SKILLS / "chemgraph/assets/calculate.py"
EVIDENCE = ("submission.started", "job.id", "job.json")
REQUIRED = ("compute_resource_id", "account", "queue", "filesystems", "node_count",
            "duration", "remote_root", "environment_setup", "python")


def prepare(run_name, workspace="."):
    workspace = Path(workspace)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        print("RUN must be a plain name such as water-opt-001.", file=sys.stderr)
        return 1
    run = workspace / run_name
    if any((run / name).exists() for name in EVIDENCE):
        print("A submission was attempted or accepted; inspect it, never regenerate it.",
              file=sys.stderr)
        return 1
    if not (run / "input.json").is_file():
        print(f"Write {run_name}/input.json first.", file=sys.stderr)
        return 1
    try:
        with open(workspace / "target.toml", "rb") as handle:
            target = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"Could not read target.toml: {error}", file=sys.stderr)
        return 1
    missing = [key for key in REQUIRED if key not in target]
    if missing:
        print(f"target.toml lacks: {', '.join(missing)}", file=sys.stderr)
        return 1
    remote = PurePosixPath(target["remote_root"]) / run_name
    if not remote.is_absolute():
        print("remote_root must be an absolute compute-node path.", file=sys.stderr)
        return 1

    launch = (TEMPLATE.read_text(encoding="utf-8")
              .replace("{{ENVIRONMENT_SETUP}}", target["environment_setup"])
              .replace("{{PYTHON}}", shlex.quote(target["python"])))
    request = {
        "resource_id": target["compute_resource_id"],
        "job_spec": {
            "name": f"cg-{run_name}",
            "executable": "/bin/bash",
            "arguments": [str(remote / "launch.sh")],
            "directory": str(remote),
            "stdout_path": str(remote / "job.out"),
            "stderr_path": str(remote / "job.err"),
            "resources": {"node_count": int(target["node_count"])},
            "attributes": {
                "duration": int(target["duration"]),
                "queue_name": target["queue"],
                "account": target["account"],
                "custom_attributes": {"filesystems": target["filesystems"]},
            },
        },
    }
    calculate = CALCULATE.read_text(encoding="utf-8")
    if (workspace / ".globus" / f"{run_name}-stage-in.json").exists() and any(
        not (run / name).is_file() or (run / name).read_text(encoding="utf-8") != text
        for name, text in (("launch.sh", launch), ("calculate.py", calculate))
    ):
        print("This run was staged with other launch files; use a new RUN instead.",
              file=sys.stderr)
        return 1
    (run / "launch.sh").write_text(launch, encoding="utf-8")
    bash = shutil.which("bash")
    if bash is None or subprocess.run([bash, "-n", str(run / "launch.sh")]).returncode:
        print("launch.sh failed bash -n (or bash is missing); fix environment_setup.",
              file=sys.stderr)
        return 1
    (run / "calculate.py").write_text(calculate, encoding="utf-8")
    text = json.dumps(request, indent=2) + "\n"
    (run / "request.json").write_text(text, encoding="utf-8")
    print(f"Wrote {run_name}/launch.sh, {run_name}/calculate.py, {run_name}/request.json:")
    print(text, end="")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run directory name inside the workspace")
    args = parser.parse_args(argv)
    return prepare(args.run)


if __name__ == "__main__":
    raise SystemExit(main())
