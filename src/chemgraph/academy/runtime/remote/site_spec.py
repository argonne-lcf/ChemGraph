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
    # submit-mode only (phase 2)
    queue: str | None = None
    walltime: str | None = None


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

    has_attach = "attach" in kv
    has_submit = "queue" in kv or "walltime" in kv
    if has_attach and has_submit:
        raise ValueError(
            f"--site {name}: cannot mix attach=... with queue=/walltime=",
        )
    if not has_attach and not has_submit:
        raise ValueError(
            f"--site {name}: specify attach=<host> OR queue=<q>,walltime=<HH:MM:SS>",
        )

    if has_submit:
        raise NotImplementedError(
            f"--site {name}: submit-mode lands in phase 2; use attach=<host> for now",
        )

    return SiteSpec(
        name=name,
        mode="attach",
        agents=agents,
        compute_host=kv["attach"],
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
    ):
        try:
            parse_site(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")

    try:
        parse_site("aurora:queue=debug;walltime=01:00:00;agents=alpha")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for submit-mode")

    print("site_spec self-check ok")
