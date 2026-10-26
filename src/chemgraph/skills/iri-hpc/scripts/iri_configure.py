"""Write WORKSPACE/target.toml for iri-hpc runs on Polaris with Eagle storage.

Resource IDs, the user's projects, and the ALCF username come from the IRI API;
the laptop's Globus collection comes from `globus endpoint local-id`. Values that
cannot be discovered are returned as questions instead of guessed. Other systems
and filesystems are configured by hand from assets/target.example.toml.

Exit codes: 0 target.toml written, 1 error or unsupported target,
2 questions for the user, printed as JSON {"ask": [...], "choices": {...}}.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tomllib
import urllib.request

from iri_submit import _OPENER, BASE_URL, TIMEOUT_S, get_token

EAGLE_COLLECTION = "05d2c76a-e867-4f67-aa57-76edeb0beda0"  # alcf#dtn_eagle, root /eagle
NEEDS = ("project", "username", "environment_setup", "python")


def _get(url, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with _OPENER.open(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as response:
        return json.load(response)


def local_collection():
    result = subprocess.run(["globus", "endpoint", "local-id"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def eagle_root(path):
    """Return PATH as /eagle/..., or None when it is not on Eagle.

    /lus/eagle/projects/P is the same directory as /eagle/P.
    """
    parts = PurePosixPath(path).parts
    if any(part in (".", "..") for part in parts):
        return None
    if parts[:4] == ("/", "lus", "eagle", "projects"):
        parts = ("/", "eagle") + parts[4:]
    if parts[:2] != ("/", "eagle") or len(parts) < 3:
        return None
    return str(PurePosixPath(*parts))


def globus_path(workspace):
    """Name WORKSPACE as Globus Connect Personal does: /~/... below home."""
    try:
        return "/~/" + workspace.relative_to(Path.home()).as_posix()
    except ValueError:
        return workspace.as_posix()


def configure(workspace, *, project=None, username=None, environment_setup=None,
              python=None, remote_root=None, base_url=BASE_URL):
    workspace = Path(workspace).resolve()
    target = workspace / "target.toml"
    if target.exists():
        print(f"{target} already exists; use it as is.", file=sys.stderr)
        return 1
    if remote_root is not None and eagle_root(remote_root) is None:
        print("Automatic setup supports Polaris with Eagle only. For other systems or "
              "filesystems, fill assets/target.example.toml by hand.", file=sys.stderr)
        return 1

    try:
        token = get_token()
    except Exception as error:  # alcf_tokens raises its own AuthError types.
        print(f"IRI login needed ({type(error).__name__}); ask the user to run: "
              "alcf-tokens login iri", file=sys.stderr)
        return 1
    try:
        resources = {(r.get("name"), r.get("resource_type")): r["id"]
                     for r in _get(f"{base_url}/status/resources")}
        projects = _get(f"{base_url}/account/projects", token)
    except (OSError, ValueError) as error:  # URLError/HTTPError are OSErrors.
        print(f"IRI discovery failed: {error}", file=sys.stderr)
        return 1
    names = sorted(p["name"] for p in projects)
    shared = set.intersection(*(set(p.get("user_ids") or ()) for p in projects)) if projects else set()

    values = {"project": project, "username": username,
              "environment_setup": environment_setup, "python": python}
    choices = {"project": names}
    if project is None and len(names) == 1:
        values["project"] = names[0]
    if project is not None and project not in names:
        print(f"Project {project!r} is not one of yours: {', '.join(names)}", file=sys.stderr)
        return 1
    if username is None and len(shared) == 1:
        values["username"] = shared.pop()
    ask = [name for name in NEEDS if not values[name]]
    if ask:
        print(json.dumps({"ask": ask, "choices": {k: v for k, v in choices.items() if k in ask}}))
        return 2

    root = eagle_root(remote_root or
                      f"/eagle/{values['project']}/{values['username']}/chemgraph-runs")
    if root is None:
        print("Project or username does not form an Eagle path.", file=sys.stderr)
        return 1
    try:
        local = local_collection()
    except (OSError, subprocess.CalledProcessError):
        print("No local Globus collection: install globus-cli, start Globus Connect "
              "Personal, and have the user run: globus login", file=sys.stderr)
        return 1
    settings = {
        "compute_resource_id": resources[("Polaris", "urn:doe-iri:resource:compute")],
        "storage_resource_id": resources[("Eagle", "urn:doe-iri:resource:storage")],
        "account": values["project"],
        "username": values["username"],
        "queue": "debug",
        "filesystems": "home:eagle",
        "node_count": 1,
        "duration": 300,
        "remote_root": root,
        "local_collection_id": local,
        "local_globus_root": globus_path(workspace),
        "remote_collection_id": EAGLE_COLLECTION,
        "remote_globus_root": root.removeprefix("/eagle"),
        "environment_setup": values["environment_setup"],
        "python": values["python"],
    }
    # JSON strings without ASCII escaping are valid TOML basic strings.
    text = "# Written by iri_configure.py for Polaris with Eagle storage.\n" + "".join(
        f"{key} = {json.dumps(value, ensure_ascii=False)}\n" for key, value in settings.items())
    if tomllib.loads(text) != settings:
        print("Generated target.toml does not round-trip; not written.", file=sys.stderr)
        return 1
    with open(target, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workspace")
    parser.add_argument("--project")
    parser.add_argument("--username", help="ALCF username")
    parser.add_argument("--environment-setup", help="shell line that activates ChemGraph on Polaris")
    parser.add_argument("--python", help="Python executable of that environment")
    parser.add_argument("--remote-root", help="parent of remote run directories, on /eagle")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    return configure(args.workspace, project=args.project, username=args.username,
                     environment_setup=args.environment_setup, python=args.python,
                     remote_root=args.remote_root, base_url=args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
