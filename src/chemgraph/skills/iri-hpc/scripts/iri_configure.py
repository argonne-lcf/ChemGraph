"""Write WORKSPACE/target.toml for a ChemGraph job on Polaris, Crux, or Aurora.

Discover resource IDs, projects, and the local Globus collection. Use the site's
shared environment unless both activation and Python overrides are supplied.

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
FLARE_COLLECTION = "f39a7a0f-5bfc-46ce-9615-ba9f8592814f"  # alcf#dtn_flare
SHARED_ENVS = {
    "polaris": "/grand/catalyst/world-shared/chemgraph/alcf_chemgraph_env",
    "crux": "/grand/catalyst/world-shared/chemgraph/crux/alcf_chemgraph_env",
    "aurora": "/lus/flare/projects/catalyst/world_shared/chemgraph/alcf_chemgraph_env",
}
NEEDS = ("project", "username")


def _get(url, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with _OPENER.open(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as response:
        return json.load(response)


def local_collection():
    result = subprocess.run(["globus", "endpoint", "local-id"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def project_path(path, filesystem):
    """Return the project-relative path for either compute mount spelling."""
    root = PurePosixPath(path)
    if ".." in root.parts:
        return None
    for mount in (f"/{filesystem}", f"/lus/{filesystem}/projects"):
        if root.is_relative_to(mount):
            relative = root.relative_to(mount)
            return relative if relative.parts else None
    return None


def globus_path(workspace):
    """Name WORKSPACE as Globus Connect Personal does: /~/... below home."""
    try:
        return "/~/" + workspace.relative_to(Path.home()).as_posix()
    except ValueError:
        return workspace.as_posix()


def configure(workspace, *, system="polaris", project=None, username=None, environment_setup=None,
              python=None, remote_root=None, base_url=BASE_URL):
    workspace = Path(workspace).resolve()
    target = workspace / "target.toml"
    if target.exists():
        print(f"{target} already exists; verify its compute resource before reuse.", file=sys.stderr)
        return 1
    filesystem = "flare" if system == "aurora" else "eagle"
    mount = "/lus/flare/projects" if filesystem == "flare" else "/eagle"
    if remote_root is not None and project_path(remote_root, filesystem) is None:
        print(f"Automatic setup for {system} requires a run directory on {filesystem}. "
              "For other storage, fill assets/target.example.toml by hand.", file=sys.stderr)
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
    try:
        compute_id = resources[(system.capitalize(), "urn:doe-iri:resource:compute")]
        storage_id = resources[(filesystem.capitalize(), "urn:doe-iri:resource:storage")]
    except KeyError as error:
        print(f"IRI discovery lacks the requested resource {error}.", file=sys.stderr)
        return 1
    names = sorted(p["name"] for p in projects)
    shared = set.intersection(*(set(p.get("user_ids") or ()) for p in projects)) if projects else set()

    if environment_setup is None and python is None:
        environment_setup = f"source {SHARED_ENVS[system]}/bin/activate"
        python = f"{SHARED_ENVS[system]}/bin/python"
    elif environment_setup is None or python is None:
        print("Pass both --environment-setup and --python, or neither for the default "
              "ChemGraph environment.", file=sys.stderr)
        return 1
    values = {"project": project, "username": username}
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

    if remote_root is None:
        remote_root = f"{mount}/{values['project']}/{values['username']}/chemgraph-runs"
    relative = project_path(remote_root, filesystem)
    if relative is None:
        print(f"Invalid project path on {filesystem}.", file=sys.stderr)
        return 1
    root = str(PurePosixPath(mount) / relative)
    try:
        local = local_collection()
    except (OSError, subprocess.CalledProcessError):
        print("No local Globus collection: install globus-cli, start Globus Connect "
              "Personal, and have the user run: globus login", file=sys.stderr)
        return 1
    filesystems = "flare" if filesystem == "flare" else "home:eagle"
    if filesystem == "eagle" and python.startswith(("/grand/", "/lus/grand/projects/")):
        filesystems += ":grand"
    settings = {
        "compute_resource_id": compute_id,
        "storage_resource_id": storage_id,
        "account": values["project"],
        "username": values["username"],
        "queue": "debug",
        # PBS holds jobs that use an undeclared filesystem, such as a /grand environment.
        "filesystems": filesystems,
        "node_count": 1,
        "duration": 300,
        "remote_root": root,
        "local_collection_id": local,
        "local_globus_root": globus_path(workspace),
        "remote_collection_id": FLARE_COLLECTION if filesystem == "flare" else EAGLE_COLLECTION,
        "remote_globus_root": f"/{relative}",
        "environment_setup": environment_setup,
        "python": python,
    }
    # JSON strings without ASCII escaping are valid TOML basic strings.
    text = f"# Written by iri_configure.py for {system} with {filesystem} storage.\n" + "".join(
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
    parser.add_argument("--system", choices=sorted(SHARED_ENVS), default="polaris")
    parser.add_argument("--project")
    parser.add_argument("--username", help="ALCF username")
    parser.add_argument("--environment-setup", help="shell line that activates ChemGraph on "
                        "the compute node (default: the shared ChemGraph environment)")
    parser.add_argument("--python", help="Python executable of that environment")
    parser.add_argument("--remote-root", help="parent of remote run directories, on the target project filesystem")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    return configure(args.workspace, system=args.system, project=args.project, username=args.username,
                     environment_setup=args.environment_setup, python=args.python,
                     remote_root=args.remote_root, base_url=args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
