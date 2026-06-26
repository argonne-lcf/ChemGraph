from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """One ``--site`` flag parsed into structured form."""

    name: str  # system profile name, e.g. "aurora"
    mode: Literal["attach", "submit"]
    agents: tuple[str, ...]
    # attach-mode only
    compute_host: str | None = None
    # submit-mode only
    queue: str | None = None
    walltime: str | None = None
    nodes: int = 1
    project: str | None = None
    filesystems: str | None = None


def parse_site(spec: str) -> SiteSpec:
    """Parse ``NAME:KEY=VAL;KEY=VAL[;...]``.

    Pairs are separated by ``;`` so that ``agents=alpha,beta,gamma``
    can use ``,`` as the within-value CSV separator without ambiguity.

    Mode is inferred: ``attach=`` -> attach; ``queue=`` -> submit; both
    or neither -> error. Phase 1 supports attach only; submit raises
    NotImplementedError.
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

    submit_keys = {"queue", "walltime", "nodes", "project", "filesystems"}
    has_attach = "attach" in kv
    has_submit = bool(submit_keys & kv.keys())
    if has_attach and has_submit:
        raise ValueError(
            f"--site {name}: cannot mix attach=... with queue=/walltime=...",
        )
    if not has_attach and not has_submit:
        raise ValueError(
            f"--site {name}: specify attach=<host> OR queue=<q>;walltime=<HH:MM:SS>",
        )

    if has_attach:
        return SiteSpec(
            name=name,
            mode="attach",
            agents=agents,
            compute_host=kv["attach"],
        )

    # submit-mode
    if "queue" not in kv:
        raise ValueError(f"--site {name}: submit-mode requires queue=<name>")
    if "walltime" not in kv:
        raise ValueError(f"--site {name}: submit-mode requires walltime=<HH:MM:SS>")
    nodes_raw = kv.get("nodes", "1")
    try:
        nodes = int(nodes_raw)
    except ValueError:
        raise ValueError(f"--site {name}: nodes must be an integer, got {nodes_raw!r}") from None
    return SiteSpec(
        name=name,
        mode="submit",
        agents=agents,
        queue=kv["queue"],
        walltime=kv["walltime"],
        nodes=nodes,
        project=kv.get("project"),
        filesystems=kv.get("filesystems"),
    )


if __name__ == "__main__":  # ponytail: assert-based self-check, no pytest
    s = parse_site("aurora:attach=x4505;agents=alpha")
    assert s.name == "aurora" and s.mode == "attach"
    assert s.compute_host == "x4505" and s.agents == ("alpha",)

    s = parse_site("aurora:agents=alpha,beta;attach=x4505")
    assert s.agents == ("alpha", "beta"), s.agents

    for bad in (
        "noseparator",
        ":onlybody=1",
        "aurora:",
        "aurora:agents=",
        "aurora:attach=x;queue=debug;walltime=01:00:00;agents=a",
        "aurora:agents=a",
        "aurora:queue=debug;agents=a",  # walltime missing
        "aurora:walltime=01:00:00;agents=a",  # queue missing
        "aurora:queue=q;walltime=1;nodes=notanint;agents=a",
    ):
        try:
            parse_site(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")

    s = parse_site("aurora:queue=debug;walltime=01:00:00;agents=alpha")
    assert s.mode == "submit" and s.queue == "debug" and s.walltime == "01:00:00"
    assert s.nodes == 1 and s.project is None and s.filesystems is None

    s = parse_site(
        "aurora:queue=debug;walltime=02:30:00;nodes=4;project=MYPROJ;"
        "filesystems=home:flare;agents=alpha,beta",
    )
    assert s.nodes == 4 and s.project == "MYPROJ" and s.filesystems == "home:flare"

    print("site_spec self-check ok")
