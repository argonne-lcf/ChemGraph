"""Phase-1 ``chemgraph academy launch`` orchestrator.

Scope: ONE attach-mode site. No preflight, no auto-bootstrap, no
multi-site, no ctrl-c affordance beyond default Popen tear-down.
Those land in phases 2-5.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from chemgraph.academy.runtime.profiles import load_system_profile
from chemgraph.academy.runtime.remote.attach_backend import (
    AttachConfig,
    start,
    stop,
    wait_ready,
)
from chemgraph.academy.runtime.remote.site_spec import SiteSpec, parse_site


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chemgraph academy launch",
        description=(
            "Phase 1: attach a single site's spawn-site to the operator's "
            "existing interactive PBS allocation via ssh. Operator must "
            "have already qsub -I'd, sourced env, and have the UAN relay "
            "running. See plan.private-local.md for the broader UX."
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
            "aurora:attach=x4505c5s0b0n0;agents=alpha,beta"
        ),
    )
    p.add_argument(
        "--bundle-root",
        required=True,
        help=(
            "Absolute path on the compute host where this ChemGraph "
            "checkout lives, e.g. /flare/ChemGraph/$USER/ChemGraph."
        ),
    )
    p.add_argument(
        "--env-script",
        help=(
            "Absolute path on the compute host of the env source script "
            "(env.{system}.sh). Defaults to {bundle-root}/env.{system}.sh."
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
        "--ready-timeout-s",
        type=float,
        default=300.0,
        help="How long to wait for agents to register (default 300s).",
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


async def _launch_one(args: argparse.Namespace) -> int:
    if len(args.site) != 1:
        print(
            "launch: phase 1 supports exactly one --site. Multi-site lands in phase 3.",
            file=sys.stderr,
        )
        return 2

    site = parse_site(args.site[0])
    if site.mode != "attach":
        # parse_site already rejects submit-mode with NotImplementedError;
        # this is defensive in case parse_site loosens later.
        print(f"launch: phase 1 supports attach-mode only (got {site.mode})", file=sys.stderr)
        return 2

    cfg = AttachConfig(
        site=site,
        run_id=args.run_id,
        campaign=args.campaign,
        bundle_root=args.bundle_root,
        env_script=_resolve_env_script(args, site),
        run_dir=_resolve_run_dir(site, args),
        exchange_type=args.exchange_type,
        http_exchange_url=args.http_exchange_url,
    )

    local_run_dir = Path(args.local_run_dir) if args.local_run_dir else Path(cfg.run_dir)

    print(
        f"[launch] site={site.name} mode=attach host={site.compute_host} "
        f"agents={','.join(site.agents)}",
        file=sys.stderr,
    )
    proc = start(cfg)
    print(f"[launch] spawn-site dispatched (ssh pid {proc.pid})", file=sys.stderr)
    try:
        registered = await wait_ready(
            cfg,
            local_run_dir=local_run_dir,
            timeout_s=args.ready_timeout_s,
        )
        print(
            f"[launch] ready: registered={sorted(registered)}. "
            f"Run is live inside YOUR allocation; exiting will leave it running. "
            f"Use 'chemgraph academy bootstrap' to kick off the campaign.",
            file=sys.stderr,
        )
        return 0
    except TimeoutError as e:
        print(f"[launch] {e}", file=sys.stderr)
        print(
            f"[launch] check the attach log on the compute host: "
            f"{cfg.run_dir}/{site.name}.attach.log",
            file=sys.stderr,
        )
        stop(proc)
        return 1
    except KeyboardInterrupt:
        print("[launch] interrupted; SIGTERM'ing remote spawn-site", file=sys.stderr)
        stop(proc)
        return 130


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_launch_one(args))


if __name__ == "__main__":
    raise SystemExit(main())
