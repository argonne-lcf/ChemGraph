"""Which OCSR models the tool can use, and which ones are actually installed.

Two separate questions, and conflating them is how you end up shipping a default
nobody can run:

  * Is a model *listed*? A registry entry means "offered", not "importable here".
  * Is it *installed*? That is :func:`chemgraph.tools.ocsr_backends.is_installed`,
    which probes the model's package instead of trusting this file.

The registry is data, not code: ``chemgraph/tools/ocsr_registry.json``. Wiring in a
new specialist means adding one entry there plus a loader in ``ocsr_backends``; the
model list, the default, and the printed listing all derive from that file. Point
``CHEMGRAPH_OCSR_REGISTRY`` at your own copy to add a model without touching the
installed package.

Accuracy figures do live here, unlike the vision-LLM side, because a caller choosing
between four specialists needs to compare them and every number comes from the same
722-image benchmark run. They are a property of those images, so they rank the four
against each other and predict nothing about yours.
"""

from __future__ import annotations

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

    # Everything below is read during module import, so a field of the wrong type
    # otherwise takes down `import chemgraph.tools.ocsr_tools` with a traceback
    # pointing at a dict comprehension in this file instead of at the bad JSON.
    for name, m in specialists.items():
        if not isinstance(m, dict):
            raise bad(f"specialist {name!r} is not an object")
        # Without this, is_installed() silently reports the model as absent and the
        # user is told to install something they already have.
        if not isinstance(m.get("import_name"), str) or not m["import_name"]:
            raise bad(f"specialist {name!r} needs an 'import_name'")
        if not isinstance(m.get("latency_s"), (int, float)) or isinstance(
            m.get("latency_s"), bool
        ):
            raise bad(f"specialist {name!r} needs a numeric 'latency_s'")
        accuracy = m.get("accuracy")
        if accuracy is not None and not isinstance(accuracy, (int, float)):
            raise bad(f"specialist {name!r} has a non-numeric 'accuracy'")
        install = m.get("install")
        if install is not None and not isinstance(install, dict):
            raise bad(f"specialist {name!r} has a non-object 'install'")

    defaults = reg.get("defaults")
    if defaults is not None and not isinstance(defaults, dict):
        raise bad("'defaults' must be an object")
    return reg


_REGISTRY = load_registry()

# Kept as a module-level name because callers and tests import it. It is a view of
# the registry file, so adding a model there adds it here.
SPECIALIST_MODELS = {
    name: {
        "import_name": m["import_name"],
        "accuracy": m.get("accuracy"),
        "latency_s": m.get("latency_s"),
        "note": m.get("note", ""),
        "install": m.get("install") or {},
    }
    for name, m in _REGISTRY["specialists"].items()
}

_DEFAULTS = _REGISTRY.get("defaults") or {}

# The model used when the caller names none. DECIMER on both of the criteria that
# matter: highest exact match of the four, and a plain pip install.
DEFAULT_SPECIALIST = _DEFAULTS.get("specialist") or next(iter(SPECIALIST_MODELS))

# What the tool's ``model=`` accepts. "llm" is the agent's own model, used when no
# specialist is installed or when the caller asks for it.
LLM_MODEL = "llm"
MODEL_CHOICES = (*SPECIALIST_MODELS, LLM_MODEL)


def describe_models(installed: list[str] | None = None,
                    measured: dict[str, dict] | None = None,
                    ready: list[str] | None = None) -> str:
    """A human-readable listing of the models, for docs and error messages.

    ``installed`` marks which ones are importable here. Pass
    :func:`chemgraph.tools.ocsr_backends.available_specialists`; it lives there
    because probing imports is not this module's job.

    ``measured`` supplies accuracies from a calibration table, so a user who refit
    on their own images is shown their own numbers. Omitting it falls back to the
    registry's figures, which were measured on the shipped benchmark.

    ``ready`` names the ones whose checkpoint is also present. Three of the four
    need a download the extra cannot do, so "installed" alone tells a user their
    setup is finished when the next call will still fail. Pass
    :func:`chemgraph.tools.ocsr_backends.usable_specialists`.
    """
    installed = set(installed or [])
    ready = set(ready) if ready is not None else installed
    measured = measured or {}
    lines = ["OCSR models (model=):"]
    for name, m in SPECIALIST_MODELS.items():
        if name not in installed:
            mark = "not installed"
        elif name in ready:
            mark = "ready"
        else:
            mark = "installed, checkpoint missing"
        default = " [default]" if name == DEFAULT_SPECIALIST else ""
        # Mark which source each number came from. A table covering a subset of the
        # installed models leaves the rest on the registry's figures, and a listing
        # mixing the two without saying so cannot be read: the model labelled most
        # accurate can show the lowest number.
        refit = (measured.get(name) or {}).get("accuracy")
        accuracy = refit if refit is not None else m.get("accuracy")
        source = " (your table)" if refit is not None else ""
        score = (f"{accuracy:.3f} exact match{source}" if accuracy is not None
                 else "unmeasured")
        latency = m.get("latency_s")
        speed = f", {latency:g}s/image" if latency else ""
        lines.append(f"  {name}{default}: {score}{speed} ({mark})")
        if m.get("note"):
            lines.append(f"      {m['note']}")
    lines.append(f"  {LLM_MODEL}: the agent's own model, no install needed")
    return "\n".join(lines)
