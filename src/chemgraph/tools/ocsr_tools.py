"""LangChain ``@tool`` wrappers for OCSR: read a structure image, return a SMILES.

The tool delegates to a plain function, so a human or a test can call the same logic
without an agent: :func:`image_to_smiles` is a ``StructuredTool`` and is not callable
directly, :func:`image_to_smiles_core` is. Same split as
:mod:`chemgraph.tools.docking_tools` over :mod:`chemgraph.tools.docking_core`.

Layering, so it is clear where to add things:

* :mod:`ocsr_core` is pure: image loading, SMILES extraction, RDKit validation.
* :mod:`ocsr_backends` runs models. Never raises; returns a narrow dict.
* this module dispatches on ``model=``, assembles the public result contract, and is
  the only place that catches everything.

Binding the LLM fallback. ``image_to_smiles`` is built by :func:`make_ocsr_tools`,
which closes over the agent's own chat model. ChemGraph cannot yet give a sub-agent a
different model, so the fallback is by definition the agent's own, and no
credential is read here.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool, tool

from chemgraph.tools import ocsr_backends as backends
from chemgraph.tools import ocsr_core as core
from chemgraph.tools import ocsr_models as models

logger = logging.getLogger(__name__)


def _unknown_model_error(model: str) -> str:
    installed = backends.available_specialists()
    return (f"unknown model {model!r}. Choose one of: "
            f"{', '.join(models.MODEL_CHOICES)}.\n\n"
            f"{models.describe_models(installed)}")


def _resolve_model(model: str | None) -> tuple[str, str]:
    """Return (name, error). ``None`` means the default, falling back to the LLM."""
    if model is None:
        default = models.DEFAULT_SPECIALIST
        if backends.is_installed(default):
            return default, ""
        # Nothing installed: the LLM is the point of the fallback, so use it rather
        # than failing with an install instruction the agent cannot act on.
        installed = backends.available_specialists()
        return (installed[0] if installed else models.LLM_MODEL), ""

    name = model.strip().lower()
    if name not in models.MODEL_CHOICES:
        return "", _unknown_model_error(model)
    return name, ""


def image_to_smiles_core(image_path: str, model: str | None = None,
                         llm: Any = None) -> dict:
    """Read a molecule's 2D structure diagram and return its SMILES.

    The plain-Python entry point; :func:`image_to_smiles` is the agent-facing wrapper.
    Never raises: every failure comes back as ``ok=False`` with an actionable
    ``error``.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, GIF or WEBP of one molecule's structure.
    model : str, optional
        ``decimer``, ``molnextr``, ``molscribe``, ``ocsrglyph``, or ``llm`` for the
        agent's own model. ``None`` picks the default specialist, or the LLM when no
        specialist is installed.
    llm : optional
        The chat model to use when ``model="llm"``. Supplied by
        :func:`make_ocsr_tools`.
    """
    name, error = _resolve_model(model)
    if error:
        return core.build_result(error=error)

    # Load and sniff before dispatching: this rejects a missing file, an oversized
    # one, and a non-image renamed to .png, and it does so identically for every
    # model instead of once per backend.
    try:
        image_bytes, mime = core.load_image_bytes(image_path)
    except (FileNotFoundError, ValueError) as e:
        return core.build_result(model_used=name, error=str(e))
    except OSError as e:
        return core.build_result(model_used=name, error=f"cannot read {image_path}: {e}")

    if name == models.LLM_MODEL:
        narrow = backends.smiles_from_llm(image_bytes, mime, llm)
    else:
        narrow = backends.smiles_from_specialist(name, image_path)

    if not narrow["ok"]:
        return core.build_result(
            model_used=narrow.get("model_used") or name,
            cold_start=narrow.get("cold_start", False),
            latency_s=narrow.get("latency_s", 0.0),
            error=narrow.get("error") or "the model returned no SMILES",
        )

    validation = core.validate_smiles_core(narrow["smiles"])
    warning = ""
    if validation.get("n_fragments", 0) > 1:
        warning = (f"the image contains {validation['n_fragments']} disconnected "
                   f"fragments (a salt, a mixture, or a reaction scheme). Ask which "
                   f"one is meant before using this SMILES.")

    return core.build_result(
        ok=True,
        smiles=narrow["smiles"],
        valid=validation.get("valid", False),
        formula=validation.get("formula"),
        n_fragments=validation.get("n_fragments", 0),
        model_used=narrow.get("model_used") or name,
        cold_start=narrow.get("cold_start", False),
        latency_s=narrow.get("latency_s", 0.0),
        warning=warning,
    )


_TOOL_DOC = """Read a molecule's 2D structure diagram from an image and return its SMILES.

    Use this when the user supplies an image file of a chemical structure.

    Models. Leave `model` unset unless the user names one; the default is the most
    accurate of the specialists that is installed. Set it when:
      - the user asked for a specific model by name;
      - the default failed on this image and you want a second opinion. Specialists
        disagree on unusual drawing styles, so one failing does not mean all will.
      - no specialist is installed: use model='llm', which reads the image with the
        agent's own model and needs no installation.

    Do NOT loop through every model hoping one succeeds. A cold model costs 5-60 s to
    load, and if the image is unreadable they usually all fail the same way. Two
    attempts is a reasonable ceiling.

    Check `valid` and `n_fragments` before acting on the answer. `valid` is false when
    RDKit could not parse what the model produced. `n_fragments` above 1 means the
    image held more than one molecule (a salt, a mixture, a reaction scheme): ask the
    user which one they meant instead of passing the SMILES to a geometry or energy
    calculation.

    No confidence number is reported. A single model cannot say how likely it is to
    be right about this particular image, and the per-model benchmark accuracy
    describes someone else's images, so quoting it here would be misleading.

    Timing. The first call for a model loads it, which cost 9-170 s when measured on
    a shared CPU node, DECIMER being the slowest. Later calls in the same process are
    0.3-5 s. `cold_start` says whether a load happened, and `latency_s` includes it
    when it did.

    Specialists are purpose-built and usually beat a general vision model on clean
    structure diagrams, while being weaker on Markush structures and reaction schemes.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, GIF or WEBP showing ONE molecule's 2D structure.
    model : str, optional
        'decimer', 'molnextr', 'molscribe', 'ocsrglyph', or 'llm'. Unset picks the
        default.

    Returns
    -------
    dict
        ok : bool
            Whether a parseable SMILES was produced.
        smiles : str or None
            The SMILES, canonicalized by RDKit.
        valid : bool
            Whether RDKit parsed it.
        formula : str or None
            Molecular formula, when valid.
        n_fragments : int
            Disconnected components; above 1 means more than one molecule.
        model_used : str
            Which model answered.
        cold_start : bool
            Whether this call paid to load the model.
        latency_s : float
            Seconds, including the load when cold_start is true.
        error : str
            Empty when ok, otherwise what went wrong and what to do about it.
        warning : str
            Non-fatal caveats, such as multiple fragments.
    """


def make_ocsr_tools(llm: Any = None) -> list:
    """Build the OCSR tools, binding ``llm`` as the vision fallback.

    The agent passes its own chat model here, which is what lets ``model='llm'`` work
    without this module resolving any credentials. Called with no model the fallback
    reports that it is unavailable instead of failing at call time inside an API
    client.
    """

    def _run(image_path: str, model: str | None = None) -> dict:
        return image_to_smiles_core(image_path, model=model, llm=llm)

    _run.__doc__ = _TOOL_DOC
    return [StructuredTool.from_function(
        func=_run,
        name="image_to_smiles",
        description=_TOOL_DOC,
    )]


@tool
def image_to_smiles(image_path: str, model: str | None = None) -> dict:
    """Read a molecule's 2D structure diagram from an image and return its SMILES.

    Module-level tool for callers that bind tools statically. It has no LLM bound, so
    `model='llm'` reports the fallback as unavailable; use `make_ocsr_tools(llm)` to
    get a version with the agent's model wired in. Every other model works here.

    See `make_ocsr_tools` for the full contract.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, GIF or WEBP showing ONE molecule's 2D structure.
    model : str, optional
        'decimer', 'molnextr', 'molscribe', 'ocsrglyph', or 'llm'.
    """
    return image_to_smiles_core(image_path, model=model, llm=None)


@tool
def list_ocsr_models() -> str:
    """List the OCSR models, their benchmark accuracy, and which are installed here.

    Use this when the user asks what models are available, or after an install error,
    so the answer names what this machine actually has instead of guessing.
    """
    return models.describe_models(backends.available_specialists())


@tool
def validate_smiles(smiles: str) -> dict:
    """Check whether a SMILES string is chemically valid, and say what is wrong.

    Use this after proposing a SMILES and before giving a final answer. If `valid` is
    false, read `errors`: it carries RDKit's own message, for example "Explicit
    valence for atom # 3 N, 4, is greater than permitted", which tells you what to
    correct. Also returns the molecular formula and atom counts so you can check them
    against the image.

    `n_fragments` greater than 1 means the string describes more than one
    disconnected molecule, which is usually a salt or two structures read as one.

    Parameters
    ----------
    smiles : str
        The candidate SMILES string.

    Returns
    -------
    dict
        smiles, valid, canonical_smiles, formula, n_atoms, n_heavy_atoms, elements,
        n_fragments, errors, rdkit_available.
    """
    return core.validate_smiles_core(smiles)
