"""``chemgraph academy launch`` orchestrator.

Phase 3: multi-site, mixed-mode, auto-bootstrap. Preflight and
status/stop subcommands are phases 4-5.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal ANSI helpers
# ---------------------------------------------------------------------------
# No dependency on rich (would pull in a lot at import time for a few
# colored lines of stderr). Auto-disable when stderr isn't a TTY OR when
# NO_COLOR is set (https://no-color.org). Operators piping to a log
# file get plain text.

_USE_COLOR = sys.stderr.isatty() and not os.environ.get("NO_COLOR")


def _ansi(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(text: str) -> str: return _ansi("32", text)
def _red(text: str) -> str: return _ansi("31", text)
def _yellow(text: str) -> str: return _ansi("33", text)
def _cyan(text: str) -> str: return _ansi("36", text)
def _bold(text: str) -> str: return _ansi("1", text)

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
            "Pass --site multiple times to bring up several HPCs at once."
        ),
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument(
        "--site",
        action="append",
        required=True,
        help=(
            "NAME:KEY=VAL;KEY=VAL... Pass multiple --site flags for "
            "multi-site launches. Examples: "
            "aurora:attach=x4505;agents=alpha  OR  "
            "crux:queue=debug;walltime=01:00:00;agents=beta;project=MYPROJ"
        ),
    )
    p.add_argument(
        "--bundle-root",
        help=(
            "Default absolute path on the compute host where this "
            "ChemGraph checkout lives, e.g. "
            "/flare/$ALCF_PROJECT/$USER/ChemGraph. Override per-site "
            "with bundle_root=... inside --site -- needed when HPCs "
            "use different filesystems (Aurora /flare vs Crux /eagle). "
            "Required unless EVERY --site supplies its own bundle_root."
        ),
    )
    p.add_argument(
        "--venv-activate",
        help=(
            "Absolute path on the compute host of the venv 'activate' "
            "script the remote bash should source so 'chemgraph' is on "
            "PATH. Defaults to "
            "$(dirname system_profile.venv_python)/activate, which "
            "matches the existing manual runbook."
        ),
    )
    p.add_argument(
        "--run-dir",
        help=(
            "Absolute path on the compute host for the run directory. "
            "Defaults to the system profile's run_root / run-id per site."
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
            "Per-site: how long to wait for agents to register, post-"
            "allocation. Default 300s. For submit-mode the queue wait "
            "gets ~80%% of this budget; bump for busy queues."
        ),
    )
    p.add_argument(
        "--auto-bootstrap",
        action="store_true",
        help=(
            "After every site reports ready, dispatch the campaign "
            "bootstrap message to the initial agent. Equivalent to "
            "running 'chemgraph academy bootstrap' manually."
        ),
    )
    p.add_argument(
        "--bootstrap-recipient",
        help=(
            "Override the bootstrap recipient (defaults to the "
            "campaign's initial_agent). Ignored without --auto-bootstrap."
        ),
    )
    p.add_argument(
        "--spawn-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "Extra argv passed through to the remote 'chemgraph academy "
            "spawn-site' invocation. Repeatable; each value is appended "
            "verbatim. Example: --spawn-arg --agents-per-node --spawn-arg 2. "
            "Use this to set spawn-site flags the launcher doesn't have "
            "a dedicated --flag for yet (agents_per_node, max_decisions, "
            "startup_timeout_s, etc)."
        ),
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "Skip the env-var preflight check. Use only when you've "
            "deliberately arranged for the remote bash to inherit env "
            "from elsewhere (e.g. login-node defaults set in /etc/profile.d)."
        ),
    )
    return p.parse_args(argv)


def _resolve_run_dir(site: SiteSpec, args: argparse.Namespace) -> str:
    if args.run_dir:
        return args.run_dir
    profile = load_system_profile(site.name)
    return str(Path(profile.run_root) / args.run_id)


def _resolve_bundle_root(site: SiteSpec, args: argparse.Namespace) -> str:
    """Per-site bundle_root wins over the global --bundle-root.

    Different HPCs may stage onto different filesystems (Aurora /flare,
    Crux /eagle), so the operator can set the common default with
    --bundle-root and override per-site inside --site.
    """
    if site.bundle_root:
        return site.bundle_root
    if args.bundle_root:
        return args.bundle_root
    raise ValueError(
        f"--site {site.name}: no bundle root resolved. Either pass "
        "--bundle-root globally or bundle_root=... inside --site.",
    )


def _resolve_venv_activate(args: argparse.Namespace, site: SiteSpec) -> str:
    """Path to the venv `activate` script the remote bash should
    source. Defaults to ``$(dirname profile.venv_python)/activate``
    which matches the existing manual runbook (operator types
    ``source .../venvs/academy-swarm/bin/activate`` before
    ``chemgraph academy spawn-site``).
    """
    if args.venv_activate:
        return args.venv_activate
    profile = load_system_profile(site.name)
    return str(Path(profile.venv_python).parent / "activate")


# Env vars the operator used to type manually inside the qsub -I
# shell before running ``chemgraph academy spawn-site``. The launcher
# now reads them from the operator's local shell and forwards them
# to the remote bash. Names are intentionally cryptic for historical
# reasons -- ALCF's account model has THREE separate identifiers for
# what looks like one user, and confusing them looks identical at a
# glance (`jinchu`, `jinchuli`, `jinchu.li` could all be the same
# person but ARE different identifiers). Comments explain each.
_FORWARDED_ENV_VARS = (
    # ALCF "project" / allocation name. Path component on /flare,
    # /eagle, /lus. Different from the OS group of the same name.
    # Example: "ChemGraph".
    "ALCF_PROJECT",
    # Workspace path username -- the directory name under the
    # project root. Compute nodes store user data under
    # ``/flare/$ALCF_PROJECT/$ALCF_USER/...``. NOT the SSH login.
    # For accounts where SSH login != path component (e.g. SSH as
    # ``jinchuli`` but workspace at ``/flare/.../jinchu/``) this is
    # the SHORTER one without the trailing characters.
    # Example: "jinchu".
    "ALCF_USER",
    # SSH login username. Used by the launcher to build the ssh
    # target (``${ALCF_SSH_USER}@aurora.alcf.anl.gov``). Defaults to
    # ALCF_USER when unset -- safe for accounts where the two match.
    # Often LONGER than ALCF_USER for accounts where they differ.
    # Example: "jinchuli".
    "ALCF_SSH_USER",
    # Argo (ALCF's gateway to internal LLM endpoints) username.
    # The part before @anl.gov in the operator's ANL email address.
    # Goes into the body of LM API calls as ``"user": "<value>"`` --
    # Argo rejects requests whose user isn't on its allowlist; this
    # is what gets put into lm_config.json's ``user`` field.
    # NOT the SSH login, NOT the workspace path component.
    # Example: "jinchu.li" (from jinchu.li@anl.gov).
    "ARGO_USER",
    # ALCF site HTTP proxy. Required on compute nodes for any
    # outbound traffic to the public internet (the hosted Academy
    # exchange at exchange.academy-agents.org). UPPERCASE variants
    # exist because some libraries read one form and not the other;
    # we propagate both to be safe. The compute_launcher's no_proxy
    # list still excludes 127.0.0.1 + .alcf.anl.gov so the local UAN
    # relay (LM traffic) keeps bypassing the proxy.
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
)


# ALCF HTTP proxy on compute nodes. Required for any outbound
# traffic from compute to the public internet (the hosted Academy
# exchange). Not parameterised because it's the same value across
# Aurora/Crux/Polaris -- and crucially, the operator's LAPTOP
# typically does NOT have http_proxy set, so we cannot rely on env
# forwarding from the operator's shell. Hardcoded so federated
# launches "just work" without operator setup.
_ALCF_HTTP_PROXY = "http://proxy.alcf.anl.gov:3128"


def _collect_remote_env(*, exchange_type: str) -> dict[str, str]:
    """Snapshot of env vars to forward to the remote bash.

    Two sources:
    1. FORWARDED env vars present in the operator's local shell
       (ALCF_*, ARGO_USER, ...). Empty values dropped so the remote
       bash doesn't end up with ``export ARGO_USER=`` and shadow
       what was inherited from the login node.
    2. When exchange_type=http (compute ranks talk to the hosted
       Academy exchange), inject the ALCF HTTP proxy + no_proxy
       list. Laptop almost never has these set, so we can't rely on
       env forwarding -- inject defaults so federated runs work
       without operator setup.
    """
    env = {k: os.environ[k] for k in _FORWARDED_ENV_VARS if os.environ.get(k)}
    if exchange_type == "http":
        # Don't overwrite operator-supplied values; the operator
        # knows their own network. Just fill in the gaps.
        env.setdefault("http_proxy", _ALCF_HTTP_PROXY)
        env.setdefault("https_proxy", _ALCF_HTTP_PROXY)
        env.setdefault("HTTP_PROXY", _ALCF_HTTP_PROXY)
        env.setdefault("HTTPS_PROXY", _ALCF_HTTP_PROXY)
        # NOTE: don't inject NO_PROXY/no_proxy. The value contains
        # commas + globs (``*.alcf.anl.gov``) which trigger four
        # layers of shell-quote escape across laptop->login->compute
        # ssh chain, producing an unparseable bash -c argument that
        # silently kills the whole script before mkdir runs (bash
        # validates the whole -c string before executing anything).
        # compute_launcher already writes the correct no_proxy from
        # profile.no_proxy via _prepare_environment, so we don't
        # need to inject it on the bash side anyway.
    return env


def _resolve_local_run_dir(
    args: argparse.Namespace,
    site: SiteSpec,
) -> Path:
    if args.local_run_dir:
        return Path(args.local_run_dir)
    return Path(_resolve_run_dir(site, args))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

# (var name, required?, concrete description, example value)
_REQUIRED_ENV_HELP = (
    (
        "ALCF_PROJECT", True,
        "Your ALCF allocation/project. The directory name right after the "
        "filesystem mount: /flare/<this>/..., /eagle/<this>/.... Check with "
        "`ls /flare` or your allocation confirmation email.",
        "ChemGraph",
    ),
    (
        "ALCF_USER", True,
        "Your folder name UNDER the project root: "
        "/flare/${ALCF_PROJECT}/<this>/... -- this is where your venv, "
        "ChemGraph checkout, and runs live. Check with "
        "`ls /flare/$ALCF_PROJECT/` from an ALCF login node.",
        "jinchu",
    ),
    (
        "ARGO_USER", True,
        "Your ANL username -- the part before @anl.gov in your "
        "ANL email address (e.g. jinchu.li@anl.gov -> ARGO_USER=jinchu.li). "
        "Goes into LM API calls as the `user` field; Argo rejects "
        "requests whose user isn't on its allowlist. NOT the ALCF SSH "
        "login, NOT the workspace folder name.",
        "jinchu.li",
    ),
    (
        "ALCF_SSH_USER", False,
        "The username before @ when you `ssh ...alcf.anl.gov` -- your "
        "ALCF SSH login. Defaults to ALCF_USER if unset; only set this "
        "explicitly when your SSH login differs from your workspace "
        "folder name (e.g. login `jinchuli` vs workspace `jinchu`).",
        "jinchuli",
    ),
)


def _preflight_env(*, stderr) -> list[str]:
    """Check operator's env for required forwarded vars. Returns a list
    of error messages (empty list = ok). Prints a per-var status
    line to stderr regardless of result so the operator can see what
    the launcher will forward.
    """
    errors: list[str] = []
    for name, required, explanation, example in _REQUIRED_ENV_HELP:
        value = os.environ.get(name, "")
        if value:
            print(
                f"[preflight] {_green('ok')}    "
                f"{_bold(name)}={_cyan(value)}",
                file=stderr,
            )
        elif required:
            print(
                f"[preflight] {_red('MISS')}  {_bold(name)}  "
                f"({explanation})",
                file=stderr,
            )
            errors.append(
                f"{name} not set -- {explanation} Example: export {name}={example}"
            )
        else:
            print(
                f"[preflight] {_yellow('skip')}  {_bold(name)}  "
                f"(optional; falls back to ALCF_USER if you don't set it)",
                file=stderr,
            )
    return errors


def _preflight_dashboard(
    pairs: "list[tuple[SiteSpec, SiteBackend]]",
    *,
    stderr,
) -> list[str]:
    """Per-site check: does the UAN relay-host file exist on the HPC?

    The dashboard launcher writes ``profile.relay_host_file`` after
    it stages and starts the UAN relay. The daemon reads that same
    file at startup to wire LM traffic. If the operator forgot to
    include this site in the dashboard's ``--system`` list, the
    daemon will crash at startup with ``Could not determine UAN
    relay host`` -- and submit-mode in particular makes this
    invisible (the launcher's wait_ready just times out polling
    placement.json that the dead daemon never wrote).

    Probe via ssh to the login node so we don't depend on the
    compute node being reachable (Crux compute isn't from outside).
    """
    from chemgraph.academy.runtime.remote.ssh_transport import (
        ssh_quote,
        ssh_run,
    )

    errors: list[str] = []
    for spec, _ in pairs:
        profile = load_system_profile(spec.name)
        login_host = profile.remote_host
        relay_file = profile.relay_host_file
        try:
            r = ssh_run(
                login_host,
                f"test -s {ssh_quote(relay_file)} && cat {ssh_quote(relay_file)} || echo MISSING",
                timeout_s=15,
                check=False,
            )
        except Exception as e:
            print(
                f"[preflight] {_yellow('warn')}  dashboard probe ssh "
                f"{spec.name} failed: {e}",
                file=stderr,
            )
            # Don't block on ssh failure -- might be temporary; let
            # the launcher fail later with a clearer message.
            continue
        stdout = (r.stdout or "").strip()
        if stdout and stdout != "MISSING":
            print(
                f"[preflight] {_green('ok')}    "
                f"dashboard {_bold(spec.name)} relay={_cyan(stdout)}",
                file=stderr,
            )
        else:
            print(
                f"[preflight] {_red('MISS')}  "
                f"dashboard {_bold(spec.name)} "
                f"(no relay-host file at {relay_file})",
                file=stderr,
            )
            errors.append(
                f"site {spec.name!r}: no UAN relay running. Did you start "
                f"the dashboard with --system including {spec.name!r}? "
                f"Try: chemgraph academy dashboard -- <run-id> --system "
                f"{','.join(sorted({s.name for s, _ in pairs}))} "
                f"--campaign <campaign>"
            )
    return errors


def _make_backend(args: argparse.Namespace, site: SiteSpec) -> SiteBackend:
    bundle_root = _resolve_bundle_root(site, args)
    run_dir = _resolve_run_dir(site, args)
    venv_activate = _resolve_venv_activate(args, site)
    local_run_dir = _resolve_local_run_dir(args, site)
    remote_env = _collect_remote_env(exchange_type=args.exchange_type)

    if site.mode == "attach":
        # On ALCF (Aurora, Crux), the laptop can't ssh directly to a
        # compute node -- compute nodes accept only login-node-
        # originated connections (hostbased). Read the login host
        # from the system profile and nest the ssh: laptop -> login
        # -> compute. profile.remote_host already encodes
        # ${ALCF_SSH_USER}@host.
        profile = load_system_profile(site.name)
        cfg = AttachConfig(
            site=site,
            run_id=args.run_id,
            campaign=args.campaign,
            bundle_root=bundle_root,
            venv_activate=venv_activate,
            run_dir=run_dir,
            remote_env=remote_env,
            login_host=profile.remote_host,
            exchange_type=args.exchange_type,
            http_exchange_url=args.http_exchange_url,
            extra_args=tuple(args.spawn_arg),
        )
        return AttachSiteBackend(cfg, local_run_dir=local_run_dir)

    profile = load_system_profile(site.name)
    cfg = SubmitConfig(
        site=site,
        run_id=args.run_id,
        campaign=args.campaign,
        login_host=profile.remote_host,
        bundle_root=bundle_root,
        env_script=venv_activate,  # submit-mode still sources whatever this points at
        run_dir=run_dir,
        exchange_type=args.exchange_type,
        http_exchange_url=args.http_exchange_url,
        project=args.project,
        extra_spawn_args=tuple(args.spawn_arg),
    )
    return SubmitSiteBackend(cfg, local_run_dir=local_run_dir)


def build_backends(
    args: argparse.Namespace,
) -> list[tuple[SiteSpec, SiteBackend]]:
    """Parse every --site flag and construct a backend per site.

    Returns the (spec, backend) pairs so the caller has both the
    parsed spec (for messaging) and the backend (for lifecycle).
    """
    out: list[tuple[SiteSpec, SiteBackend]] = []
    for raw in args.site:
        spec = parse_site(raw)
        out.append((spec, _make_backend(args, spec)))
    return out


async def _stop_all(backends: list[SiteBackend], *, force: bool = False) -> None:
    """Best-effort tear-down. Each backend swallows its own exceptions
    inside stop() already; we still wrap with return_exceptions to
    prevent one failing stop from leaving siblings behind."""
    await asyncio.gather(
        *(b.stop(force=force) for b in backends),
        return_exceptions=True,
    )


def _run_bootstrap(args: argparse.Namespace) -> int:
    """Invoke bootstrap subcommand in-process (it ships its own
    argv-based main, so we just hand it the right argv).
    """
    from chemgraph.academy.runtime.bootstrap import main as bootstrap_main

    bs_argv = [
        "--campaign", args.campaign,
        "--run-id", args.run_id,
        "--exchange-type", args.exchange_type,
    ]
    if args.http_exchange_url:
        bs_argv += ["--http-exchange-url", args.http_exchange_url]
    if args.bootstrap_recipient:
        bs_argv += ["--recipient", args.bootstrap_recipient]
    return bootstrap_main(bs_argv) or 0


async def _launch(args: argparse.Namespace) -> int:
    # Preflight: catch missing operator env vars (ALCF_USER, ARGO_USER,
    # ...) at launch time instead of letting them surface as cryptic
    # daemon-side failures 30s into the run. Skip via --skip-preflight
    # for the edge case where the operator deliberately wants to defer
    # to login-node defaults.
    if not args.skip_preflight:
        errors = _preflight_env(stderr=sys.stderr)
        if errors:
            print(
                f"[preflight] {_red('FAIL')} -- fix the missing env "
                "vars (or pass --skip-preflight if you know what you're "
                "doing):",
                file=sys.stderr,
            )
            for msg in errors:
                print(f"  - {msg}", file=sys.stderr)
            return 2

    try:
        pairs = build_backends(args)
    except (ValueError, NotImplementedError) as e:
        print(f"launch: {e}", file=sys.stderr)
        return 2

    # Per-site dashboard-presence probe. Cheap (one ssh per site) and
    # catches the "operator forgot --system on dashboard" failure
    # mode that otherwise surfaces as a silent timeout deep in the
    # daemon. Same --skip-preflight escape hatch as the env check.
    if not args.skip_preflight:
        dash_errors = _preflight_dashboard(pairs, stderr=sys.stderr)
        if dash_errors:
            print(
                f"[preflight] {_red('FAIL')} -- dashboard isn't set "
                "up for one or more sites:",
                file=sys.stderr,
            )
            for msg in dash_errors:
                print(f"  - {msg}", file=sys.stderr)
            return 2

    backends = [b for _, b in pairs]
    site_summary = ", ".join(
        f"{spec.name}({spec.mode})" for spec, _ in pairs
    )
    print(f"[launch] sites=[{site_summary}] run-id={args.run_id}", file=sys.stderr)

    # Phase: start every site in parallel. If any fails to start, tear
    # down whatever did start before propagating.
    start_results = await asyncio.gather(
        *(b.start() for b in backends),
        return_exceptions=True,
    )
    start_errors = [
        (spec, exc)
        for (spec, _), exc in zip(pairs, start_results)
        if isinstance(exc, BaseException)
    ]
    if start_errors:
        for spec, exc in start_errors:
            print(f"[launch] start failed for {spec.name}: {exc}", file=sys.stderr)
        await _stop_all(backends)
        return 1

    # Phase: wait for every site to report ready in parallel. asyncio
    # cancels the in-flight waiters on exception, so the first miss
    # short-circuits the rest. The wait_ready coroutines own their
    # own timeouts.
    try:
        ready_results = await asyncio.gather(
            *(
                b.wait_ready(
                    local_run_dir=_resolve_local_run_dir(args, spec),
                    timeout_s=args.ready_timeout_s,
                )
                for spec, b in pairs
            ),
        )
    except (TimeoutError, RuntimeError) as e:
        print(f"[launch] wait_ready failed: {e}", file=sys.stderr)
        await _stop_all(backends)
        return 1
    except KeyboardInterrupt:
        print("[launch] interrupted during wait_ready; tearing down", file=sys.stderr)
        await _stop_all(backends)
        return 130

    for (spec, _), agents in zip(pairs, ready_results):
        print(
            f"[launch] {_green('ready')}: {_bold(spec.name)} -> "
            f"{sorted(agents)}",
            file=sys.stderr,
        )

    # Phase: auto-bootstrap.
    if args.auto_bootstrap:
        print(
            f"[launch] all sites {_green('ready')}, dispatching bootstrap...",
            file=sys.stderr,
        )
        # bootstrap_main is blocking and runs its own asyncio loop;
        # offload to a thread so a future operator-side keepalive in
        # this coroutine doesn't deadlock against it.
        rc = await asyncio.to_thread(_run_bootstrap, args)
        if rc:
            print(
                f"[launch] {_red('bootstrap failed')} (exit {rc})",
                file=sys.stderr,
            )
            # Don't tear down -- agents are still useful, operator can
            # re-run bootstrap manually after fixing the cause.
            return rc
        print(
            f"[launch] {_green('bootstrap dispatched')}.",
            file=sys.stderr,
        )

    # Phase: report and exit. The compute processes keep running --
    # attach-mode lives inside the operator's allocation, submit-mode
    # has its own PBS job. Either way, exiting the launcher does not
    # tear them down.
    print(
        "[launch] launch complete. Compute processes continue running.",
        file=sys.stderr,
    )
    for spec, _ in pairs:
        if spec.mode == "attach":
            note = "inside YOUR allocation; exiting your interactive shell will end it"
        else:
            note = "as a PBS job; qdel manually to cancel"
        print(f"[launch]   {spec.name}: {spec.mode}-mode -- {note}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_launch(args))


if __name__ == "__main__":
    raise SystemExit(main())
