#!/usr/bin/env python3
"""Remove only resources carrying one exact ChemGraph evaluation run label."""

import argparse
import json
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", required=True, help="run_id from OUTPUT_DIR/podman-run.json or a cleanup error")
args = parser.parse_args()
if not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
    parser.error("run-id must be exactly 32 lowercase hexadecimal characters")
label = f"org.chemgraph.evaluation.run={args.run_id}"
for kind in ("container", "volume", "network"):
    command = ["podman", "--remote=false", kind]
    result = subprocess.run(command + ["ls", *(["--all"] if kind == "container" else []),
                                      "--filter", f"label={label}", "--format=json"],
                            check=True, capture_output=True, text=True)
    for row in json.loads(result.stdout):
        names = row.get("Names") or []
        name = row.get("Name") or row.get("name") or (names[0] if names else None)
        if not name or not name.startswith(f"cg-eval-{args.run_id}-"):
            raise RuntimeError("Unexpected resource name; refusing cleanup")
        subprocess.run(command + ["rm", *(["--force"] if kind == "container" else []), name], check=True)
