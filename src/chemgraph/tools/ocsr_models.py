"""Which backends and models the OCSR tool can use, and which ones actually work.

Two separate questions, and conflating them is how you end up shipping a default that
nobody can run:

  * Is a model *listed*? A registry entry means "offered", not "healthy right now".
  * Does it *see the image*? A text-only model handed a picture does not say
    "unsupported"; it confidently describes an image it never saw. The only way to
    know is to ask it about a picture and check the answer.

So the tables below carry a measured status, not just a name, and
:func:`describe_backends` prints them for a user choosing a backend. Availability
changes, so probe an endpoint before relying on it instead of trusting a
date-stamped note here.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# The registry is data, not code: chemgraph/tools/ocsr_registry.json.
#
# Wiring in a new specialist means adding one entry there and dropping its worker
# script into ocsr_workers/. Nothing in this module needs editing -- the backend name,
# the installer, ensemble membership, and the printed listing are all derived from
# that file. Point CHEMGRAPH_OCSR_REGISTRY at your own copy to add a model without
# touching the installed package at all.
#
# No accuracy figures live here, for either family.
#
# Specialist accuracy belongs in the calibration table, beside the k, n and interval
# that produced it, because it feeds the confidence a caller acts on and a refit has
# to update every reported number at once.
#
# Vision LLMs get no accuracy at all. People use ChemGraph to run their own
# benchmarks, on their own images, and a figure shipped here would answer the
# question they came to ask, from a run they did not do and cannot inspect. What the
# registry records instead is what a user cannot find out without trying: whether an
# endpoint answers, and whether the model was seen to drive a tool call.
# --------------------------------------------------------------------------- #
_REGISTRY_ENV = "CHEMGRAPH_OCSR_REGISTRY"


def load_registry(path: str | None = None) -> dict:
    """Load the model registry: explicit path, then env var, then the packaged file."""
    import json
    import os

    candidate = path or os.environ.get(_REGISTRY_ENV)
    if candidate:
        with open(os.path.expanduser(candidate)) as fh:
            return _validate_registry(json.load(fh), os.path.expanduser(candidate))

    from importlib import resources

    ref = resources.files("chemgraph.tools").joinpath("ocsr_registry.json")
    with ref.open() as fh:
        return _validate_registry(json.load(fh), "packaged registry")


def _validate_registry(reg: object, origin: str) -> dict:
    """Reject a registry that would fail later in a confusing place."""
    def bad(why: str) -> ValueError:
        return ValueError(f"OCSR registry {origin!r} is unusable: {why}")

    if not isinstance(reg, dict):
        raise bad(f"top level is {type(reg).__name__}, expected an object")
    specialists = reg.get("specialists")
    if not isinstance(specialists, dict) or not specialists:
        raise bad("'specialists' must be a non-empty object")
    # Everything below is read during module import or by describe_backends, so a
    # field of the wrong type takes down `import chemgraph.tools.ocsr_tools` with a
    # traceback pointing at a dict comprehension in this file instead of at the line
    # of JSON the user got wrong.
    for name, m in specialists.items():
        if not isinstance(m, dict):
            raise bad(f"specialist {name!r} is not an object")
        # Without a worker script the backend resolves and then dies at call time
        # with a missing-file error that names an internal path, not the model.
        if not isinstance(m.get("worker"), str) or not m["worker"]:
            raise bad(f"specialist {name!r} needs a 'worker' script filename")
        if not isinstance(m.get("latency_s"), (int, float)) or isinstance(
            m.get("latency_s"), bool
        ):
            raise bad(f"specialist {name!r} needs a numeric 'latency_s'")
        install = m.get("install")
        if install is not None and not isinstance(install, dict):
            raise bad(f"specialist {name!r} has a non-object 'install'")
    llms = reg.get("vision_llms") or {}
    if not isinstance(llms, dict):
        raise bad("'vision_llms' must be an object")
    for endpoint in ("alcf", "shim"):
        entries = llms.get(endpoint)
        if entries is None:
            continue
        if not isinstance(entries, dict):
            raise bad(f"'vision_llms.{endpoint}' must be an object")
        for name, m in entries.items():
            if not isinstance(m, dict):
                raise bad(f"vision_llms.{endpoint}.{name} is not an object")
            if endpoint == "shim" and not isinstance(m.get("wire_name"), str):
                raise bad(
                    f"shim model {name!r} needs a 'wire_name': the shim rejects the "
                    f"friendly spelling, so without it every call fails on the wire"
                )
    defaults = reg.get("defaults")
    if defaults is not None and not isinstance(defaults, dict):
        raise bad("'defaults' must be an object")
    return reg


_REGISTRY = load_registry()

# Kept as module-level names because callers and tests already import them. Each is a
# view of the registry file, so adding a model there adds it here.
SPECIALIST_MODELS = {
    name: {"latency_s": m.get("latency_s"), "note": m.get("note", "")}
    for name, m in _REGISTRY["specialists"].items()
}

ALCF_VISION_MODELS = {
    name: {"note": m.get("note", "")}
    for name, m in (_REGISTRY.get("vision_llms", {}).get("alcf") or {}).items()
}

# Keys are the friendly spelling a user types; values are the wire name the shim
# requires. Sending one the other's spelling fails mid-run with "Invalid model".
SHIM_VISION_MODELS = {
    name: m["wire_name"]
    for name, m in (_REGISTRY.get("vision_llms", {}).get("shim") or {}).items()
}

_DEFAULTS = _REGISTRY.get("defaults") or {}

# The default specialist must exist. Falling back to the first registered model rather
# than to a name baked in here keeps a registry that drops 'decimer' working.
DEFAULT_SPECIALIST = _DEFAULTS.get("specialist")
if DEFAULT_SPECIALIST not in SPECIALIST_MODELS:
    DEFAULT_SPECIALIST = next(iter(SPECIALIST_MODELS))

DEFAULT_ALCF_MODEL = _DEFAULTS.get("alcf") or next(iter(ALCF_VISION_MODELS), None)
DEFAULT_SHIM_MODEL = _DEFAULTS.get("shim") or next(iter(SHIM_VISION_MODELS), None)

# Every registered specialist is also a backend name, so adding one to the registry
# makes backend="<name>" work with no change here.
BACKENDS = {
    "auto": f"One fast, accurate specialist ({DEFAULT_SPECIALIST}). The default.",
    "ensemble": (
        f"All {len(SPECIALIST_MODELS)} specialists, voted, with a calibrated confidence."
    ),
    **{name: "That specialist alone." for name in SPECIALIST_MODELS},
    "alcf": "A vision LLM on ALCF. Pick one with model=.",
    "shim": "A vision LLM through the local Argo shim. Pick one with model=.",
    "llm": "Whichever of alcf/shim is reachable; reports which in backend_used.",
}


def resolve_model(backend: str, model: str | None) -> str:
    """Return the model name to send on the wire for a backend.

    Accepts the friendly Argo spelling and translates it, so a caller never has to
    remember that the shim wants ``claudeopus48`` rather than ``claude-opus-4.8``.

    Raises
    ------
    ValueError
        If the model is not known to be vision-capable on that backend. Refusing here
        is deliberate: a text-only model given an image returns fluent text about a
        picture it never saw, which is worse than an error.
    """
    if backend == "alcf":
        name = model or DEFAULT_ALCF_MODEL
        if name not in ALCF_VISION_MODELS:
            raise ValueError(
                f"{name!r} is not a known ALCF vision model. Choose from: "
                f"{', '.join(ALCF_VISION_MODELS)}"
            )
        return name
    if backend == "shim":
        name = model or DEFAULT_SHIM_MODEL
        if name in SHIM_VISION_MODELS:
            return SHIM_VISION_MODELS[name]
        if name in SHIM_VISION_MODELS.values():
            return name  # already a wire name
        raise ValueError(
            f"{name!r} is not a known vision model on the Argo shim. Choose from: "
            f"{', '.join(SHIM_VISION_MODELS)}"
        )
    if backend in SPECIALIST_MODELS:
        return backend  # the backend name IS the model
    raise ValueError(f"backend {backend!r} takes no model= argument")


def describe_backends(calibration: str | None = None) -> str:
    """Human-readable table of every backend and model, for the CLI and the docs.

    Specialist accuracies are read from the calibration table, so this listing tracks
    whatever table is in force rather than repeating figures compiled into the source.
    Point it at a custom table to see that table's numbers.
    """
    from chemgraph.tools.ocsr_core import load_calibration

    try:
        table = load_calibration(calibration)
    except (OSError, ValueError, TypeError):
        table = {}
    perf = table.get("model_performance") or {}
    n_items = table.get("n_items")
    committee = table.get("committee") or list(SPECIALIST_MODELS)

    lines = ["OCSR backends (image -> SMILES)", ""]
    lines.append("  Specialist models, local, no network:")
    for name, m in SPECIALIST_MODELS.items():
        acc = perf.get(name, {}).get("accuracy")
        shown = f"{acc:.1%} exact" if acc is not None else "unmeasured "
        lines.append(f"    {name:12s} {shown:12s} {m['latency_s']:>5.2f}s  {m['note']}")
    ensemble = ", ".join(committee)
    lines += [
        "",
        "  Ensemble:",
        f"    ensemble     {ensemble}, voted, with a calibrated confidence",
        "",
        "  Vision LLMs on ALCF   (backend='alcf', model=<name>):",
    ]
    for name, m in ALCF_VISION_MODELS.items():
        lines.append(f"    {name}")
        # No accuracy column: whether the endpoint answers is the thing a user cannot
        # discover without trying, and how well it reads their images is what they
        # came here to measure.
        if m.get("note"):
            lines.append(f"      {m['note']}")
    lines += ["", "  Vision LLMs via the Argo shim   (backend='shim', model=<name>):"]
    shim_entries = _REGISTRY.get("vision_llms", {}).get("shim") or {}
    for friendly, wire in SHIM_VISION_MODELS.items():
        note = (shim_entries.get(friendly) or {}).get("note", "")
        lines.append(f"    {friendly:28s} (sent as {wire:16s}) {note}".rstrip())
    scope = f" on {n_items} images" if n_items else ""
    lines += [
        "",
        f"  Specialist accuracy is exact-match, stereo-blind{scope}, read from the",
        "  calibration table in force. It describes the model overall, so it is not",
        "  reported as a confidence: only backend='ensemble' gives a number for the",
        "  image in front of you.",
        "",
        "  No accuracy is quoted for the vision LLMs. Measure them on your own images",
        "  with chemgraph.tools.ocsr_calibrate; a number from someone else's run",
        "  answers a question you are better off asking of your own data.",
        "  Specialist latency is warm; the first call also loads the model, which is",
        "  5-25 s for most and 50-66 s for DECIMER (measured).",
        "  A listed LLM may still be cold or down: probe before relying on one.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_backends())
