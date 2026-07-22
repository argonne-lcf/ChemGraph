from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """One ``--site`` flag parsed into structured form."""

    name: str  # system profile name, e.g. "aurora"
    agents: tuple[str, ...]
    # per-site override for the launcher's global --bundle-root; needed
    # when HPCs use different filesystems (Aurora /flare vs Crux /eagle).
    bundle_root: str | None = None
    queue: str | None = None
    walltime: str | None = None
    nodes: int = 1
    project: str | None = None
    filesystems: str | None = None


def parse_site(spec: str) -> SiteSpec:
    """Parse ``NAME:KEY=VAL;KEY=VAL[;...]``.

    Pairs are separated by ``;`` so ``agents=alpha,beta`` can use ``,``
    as the within-value CSV separator without ambiguity. Only submit-mode
    is supported: ``queue=`` and ``walltime=`` are required.
    """
    if ":" not in spec:
        raise ValueError(f"--site must be NAME:KEY=VAL;...  got: {spec!r}")
    name, body = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--site is missing a name: {spec!r}")

    kv: dict[str, str] = {}
    for part in body.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--site key without value: {part!r} in {spec!r}")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()

    if "agents" not in kv or not kv["agents"]:
        raise ValueError(f"--site {name}: missing agents=<csv>")
    agents = tuple(a.strip() for a in kv["agents"].split(",") if a.strip())

    if "queue" not in kv:
        raise ValueError(f"--site {name}: requires queue=<name>")
    if "walltime" not in kv:
        raise ValueError(f"--site {name}: requires walltime=<HH:MM:SS>")
    nodes_raw = kv.get("nodes", "1")
    try:
        nodes = int(nodes_raw)
    except ValueError:
        raise ValueError(f"--site {name}: nodes must be an integer, got {nodes_raw!r}") from None

    return SiteSpec(
        name=name,
        agents=agents,
        queue=kv["queue"],
        walltime=kv["walltime"],
        nodes=nodes,
        project=kv.get("project"),
        filesystems=kv.get("filesystems"),
        bundle_root=kv.get("bundle_root"),
    )


if __name__ == "__main__":  # ponytail: assert-based self-check, no pytest
    s = parse_site("aurora:queue=debug;walltime=01:00:00;agents=alpha")
    assert s.name == "aurora" and s.queue == "debug" and s.walltime == "01:00:00"
    assert s.nodes == 1 and s.project is None and s.filesystems is None
    assert s.agents == ("alpha",)

    s = parse_site(
        "aurora:queue=debug;walltime=02:30:00;nodes=4;project=MYPROJ;"
        "filesystems=home:flare;agents=alpha,beta",
    )
    assert s.nodes == 4 and s.project == "MYPROJ" and s.filesystems == "home:flare"
    assert s.agents == ("alpha", "beta")

    for bad in (
        "noseparator",
        ":onlybody=1",
        "aurora:",
        "aurora:agents=",
        "aurora:agents=a",  # missing queue+walltime
        "aurora:queue=debug;agents=a",  # missing walltime
        "aurora:walltime=01:00:00;agents=a",  # missing queue
        "aurora:queue=q;walltime=1;nodes=notanint;agents=a",
    ):
        try:
            parse_site(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")

    print("site_spec self-check ok")
