"""``chemgraph academy launch`` orchestrator.

Phase 2: one --site, attach OR submit. Multi-site, preflight,
auto-bootstrap, status/stop are phases 3-5.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from chemgraph.academy.runtime.profiles import load_system_profile
from chemgraph.academy.runtime.remote.attach_backend import (
    AttachConfig,
    AttachSiteBackend,
)
from chemgraph.academy.runtime.remote.site_backend import SiteBackend
from chemgraph.academy.runtime.remote.site_spec import SiteSpec, parse_site
from chemgraph.academy.runtime.remote.submit_backend import (
    SubmitConfig,
    SubmitSiteBackend,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chemgraph academy launch",
        description=(
            "Drive a federated launch from the operator's laptop. "
            "Each --site is either attach-mode (ssh straight into a "
            "compute node inside an existing interactive PBS allocation) "
            "or submit-mode (qsub a fresh PBS job via the login node). "
            "Phase 2 still supports only one --site; multi-site is phase 3."
        ),
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument(
        "--site",
        action="append",
        required=True,
        help=(
            "NAME:KEY=VAL;KEY=VAL... e.g. "
            "aurora:attach=x4505;agents=alpha OR "
            "aurora:queue=debug;walltime=01:00:00;agents=alpha;project=MYPROJ"
        ),
    )
    p.add_argument(
        "--bundle-root",
        required=True,
        help=(
            "Absolute path on the compute host where this ChemGraph "
            "checkout lives, e.g. /flare/$ALCF_PROJECT/$USER/ChemGraph."
        ),
    )
    p.add_argument(
        "--env-script",
        help=(
            "Absolute path on the compute host of the env source script. "
            "Defaults to {bundle-root}/env.{system}.sh."
        ),
    )
    p.add_argument(
        "--run-dir",
        help=(
            "Absolute path on the compute host for the run directory. "
            "Defaults to the system profile's run_root / run-id."
        ),
    )
    p.add_argument(
        "--local-run-dir",
        help=(
            "Local mirror of the run dir (rsync'd by the dashboard). "
            "If present, placement.json is polled locally first."
        ),
    )
    p.add_argument(
        "--exchange-type",
        default="http",
        help="Forwarded to spawn-site (default: http).",
    )
    p.add_argument("--http-exchange-url")
    p.add_argument(
        "--project",
        help=(
            "PBS -A project for submit-mode. May also be set via "
            "project=... inside the --site flag (per-site wins). "
            "Ignored for attach-mode."
        ),
    )
    p.add_argument(
        "--ready-timeout-s",
        type=float,
        default=300.0,
        help=(
            "How long to wait for agents to register, post-allocation. "
            "Default 300s. For submit-mode the queue wait gets ~80%% of "
            "this budget; bump higher when queueing into busy queues."
        ),
    )
    return p.parse_args(argv)


def _resolve_run_dir(site: SiteSpec, args: argparse.Namespace) -> str:
    if args.run_dir:
        return args.run_dir
    profile = load_system_profile(site.name)
    return str(Path(profile.run_root) / args.run_id)


def _resolve_env_script(args: argparse.Namespace, site: SiteSpec) -> str:
    if args.env_script:
        return args.env_script
    return f"{args.bundle_root.rstrip('/')}/env.{site.name}.sh"


def _make_backend(args: argparse.Namespace, site: SiteSpec) -> SiteBackend:
    run_dir = _resolve_run_dir(site, args)
    env_script = _resolve_env_script(args, site)
    local_run_dir = Path(args.local_run_dir) if args.local_run_dir else Path(run_dir)

    if site.mode == "attach":
        cfg = AttachConfig(
            site=site,
            run_id=args.run_id,
            campaign=args.campaign,
            bundle_root=args.bundle_root,
            env_script=env_script,
            run_dir=run_dir,
            exchange_type=args.exchange_type,
            http_exchange_url=args.http_exchange_url,
        )
        return AttachSiteBackend(cfg, local_run_dir=local_run_dir)

    # submit-mode: ssh target comes from the system profile's remote_host
    # (it already encodes $ALCF_SSH_USER@host correctly).
    profile = load_system_profile(site.name)
    cfg = SubmitConfig(
        site=site,
        run_id=args.run_id,
        campaign=args.campaign,
        login_host=profile.remote_host,
        bundle_root=args.bundle_root,
        env_script=env_script,
        run_dir=run_dir,
        exchange_type=args.exchange_type,
        http_exchange_url=args.http_exchange_url,
        project=args.project,
    )
    return SubmitSiteBackend(cfg)


async def _launch_one(args: argparse.Namespace) -> int:
    if len(args.site) != 1:
        print(
            "launch: phase 2 supports exactly one --site. Multi-site lands in phase 3.",
            file=sys.stderr,
        )
        return 2

    site = parse_site(args.site[0])
    try:
        backend = _make_backend(args, site)
    except ValueError as e:
        print(f"launch: {e}", file=sys.stderr)
        return 2

    local_run_dir = (
        Path(args.local_run_dir)
        if args.local_run_dir
        else Path(_resolve_run_dir(site, args))
    )

    print(
        f"[launch] site={site.name} mode={site.mode} agents={','.join(site.agents)}",
        file=sys.stderr,
    )
    try:
        await backend.start()
        registered = await backend.wait_ready(
            local_run_dir=local_run_dir,
            timeout_s=args.ready_timeout_s,
        )
        mode_note = (
            "run is live inside YOUR allocation; exiting will leave it running."
            if site.mode == "attach"
            else "PBS job continues running."
        )
        print(
            f"[launch] ready: registered={sorted(registered)}. {mode_note} "
            f"Use 'chemgraph academy bootstrap' to kick off the campaign.",
            file=sys.stderr,
        )
        return 0
    except TimeoutError as e:
        print(f"[launch] {e}", file=sys.stderr)
        await backend.stop()
        return 1
    except RuntimeError as e:
        print(f"[launch] {e}", file=sys.stderr)
        await backend.stop()
        return 1
    except KeyboardInterrupt:
        print(
            f"[launch] interrupted; cancelling {site.mode}-mode work",
            file=sys.stderr,
        )
        await backend.stop()
        return 130


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_launch_one(args))


if __name__ == "__main__":
    raise SystemExit(main())
