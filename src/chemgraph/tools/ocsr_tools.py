"""LangChain ``@tool`` wrappers for OCSR: read a structure image, return a SMILES.

Each tool delegates to a plain function so a human or a test can call the same logic
without an agent: ``image_to_smiles`` is a ``StructuredTool`` and is not callable
directly, ``image_to_smiles_core`` is. Same split as
:mod:`chemgraph.tools.docking_tools` over :mod:`chemgraph.tools.docking_core`.

Layering, so it is clear where to add things:

* :mod:`ocsr_core` is pure: validation, canonicalization, voting, the confidence table.
  No network, no subprocesses.
* :mod:`ocsr_backends` runs models. Never raises; returns a narrow dict.
* this module dispatches on ``backend=``, assembles the public 18-key contract, and
  is the only place that catches everything.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from chemgraph.tools import ocsr_backends as backends
from chemgraph.tools import ocsr_core as core
from chemgraph.tools import ocsr_models as models

logger = logging.getLogger(__name__)

# Derived from the registry, so a specialist added to ocsr_registry.json becomes a
# usable backend here without an edit. Read at call time rather than bound once, so a
# test or a user pointing CHEMGRAPH_OCSR_REGISTRY at another file is honoured.
_LLM_BACKENDS = ("alcf", "shim", "llm")


def _specialists() -> tuple[str, ...]:
    return tuple(models.SPECIALIST_MODELS)


def _default_single() -> str:
    return models.DEFAULT_SPECIALIST


def _prior_with(model_name: str, calibration: str | None) -> dict:
    """A single model's prior, read from the table the caller named.

    Without threading `calibration` through, prior_confidence loads the packaged
    default, so a user who refit on their own images got their own numbers from the
    ensemble and someone else's from every single-model backend, the 'auto' default
    included, with nothing in the result to show it.

    An unreadable path is reported rather than silently falling back, since falling
    back would answer with the very numbers the caller was trying to replace.
    """
    try:
        table = core.load_calibration(calibration)
    except (OSError, ValueError, TypeError) as exc:
        return {"p": None, "label": "unavailable",
                "reason": f"calibration_unreadable: {type(exc).__name__}"}
    return core.prior_confidence(model_name, table)


def _finish(narrow: dict, *, agreement: str, basis: str | None,
            backend_used: str, conf: dict | None = None,
            votes: dict | None = None, abstained: dict | None = None) -> dict:
    """Turn a backend's narrow dict plus a confidence verdict into the contract.

    One place assembles the 18 keys, so a backend cannot ship a partial result and an
    agent never has to test for a missing field. ``raw`` is dropped here rather than
    returned: it is useful in a log, not in an agent's context.
    """
    if not narrow["ok"]:
        return core.build_result(
            ok=False, error=narrow["error"], backend_used=backend_used,
            model_used=narrow["model_used"], cold_start=narrow["cold_start"],
            latency_s=narrow["latency_s"], agreement=agreement,
            confidence_unavailable_reason="no_prediction",
        )

    v = core.validate_smiles_core(narrow["smiles"])
    warning = ""
    if v["n_fragments"] > 1:
        warning = (f"this image gave {v['n_fragments']} disconnected molecules; ask "
                   f"which one is meant before using it for a calculation")

    conf = conf or {"p": None, "label": "unavailable", "reason": "no_prior_for_model"}
    return core.build_result(
        ok=True,
        smiles=narrow["smiles"],
        valid=v["valid"],
        formula=v["formula"],
        n_fragments=v["n_fragments"],
        confidence=conf["p"],
        confidence_label=conf["label"],
        confidence_unavailable_reason=conf.get("reason"),
        agreement=agreement,
        basis=basis,
        backend_used=backend_used,
        model_used=narrow["model_used"],
        cold_start=narrow["cold_start"],
        latency_s=narrow["latency_s"],
        warning=warning,
        votes=votes,
        abstained=abstained,
    )


def _run_ensemble(image_bytes: bytes, calibration: str | None,
                  models_wanted: list[str] | None = None) -> dict:
    """Vote a committee of specialists and attach a calibrated confidence.

    Runs every installed specialist unless ``models_wanted`` names a subset. A subset
    is worth supporting because the committee is what a table describes: someone who
    has all four installed but fitted a table on two of them can only get a number by
    running exactly those two.
    """
    installed = backends.available_specialists()
    if not installed:
        return core.build_result(
            ok=False, backend_used="none",
            error=backends._setup_hint(),
            confidence_unavailable_reason="no_specialists_installed",
        )

    if models_wanted is not None:
        # A bare string iterates per character, and a non-string element blows up the
        # join below with a TypeError that names neither the argument nor the valid
        # names. An empty list is a caller who filtered down to nothing, which must
        # not silently become "run everything".
        if isinstance(models_wanted, str) or not isinstance(
            models_wanted, (list, tuple, set, frozenset)
        ):
            return core.build_result(
                ok=False, backend_used="ensemble",
                error=(f"models_wanted must be a list of specialist names, got "
                       f"{type(models_wanted).__name__}. Choose from: "
                       f"{', '.join(_specialists())}"),
            )
        models_wanted = list(models_wanted)
        if not models_wanted:
            return core.build_result(
                ok=False, backend_used="ensemble",
                error=(f"models_wanted is empty. Omit it to vote every installed "
                       f"specialist, or name some of: {', '.join(_specialists())}"),
            )
        unknown = [m for m in models_wanted if m not in _specialists()]
        if unknown:
            return core.build_result(
                ok=False, backend_used="ensemble",
                error=(f"not OCSR specialists: "
                       f"{', '.join(repr(m) for m in unknown)}. Choose from: "
                       f"{', '.join(_specialists())}"),
            )
        absent = [m for m in models_wanted if m not in installed]
        if absent:
            return core.build_result(
                ok=False, backend_used="ensemble",
                error=(f"requested but not installed: {', '.join(absent)}. Install "
                       f"with: " + "; ".join(
                           f"python -m chemgraph.tools.ocsr_setup {m}"
                           for m in absent)),
                confidence_unavailable_reason="no_specialists_installed",
            )
        installed = [m for m in installed if m in models_wanted]

    results, cold, total = [], False, 0.0
    for name in installed:
        r = backends.smiles_from_specialist(name, image_bytes)
        cold = cold or r["cold_start"]
        total += r["latency_s"]
        results.append({"model": name, "smiles": r["smiles"],
                        "ok": r["ok"], "error": r["error"], "infer_s": r["latency_s"]})

    # Load before voting: the table records the model priority it was fitted under,
    # and that order decides which answer wins a tie. Voting by the registry's order
    # instead would attach a number measured for one model's answer to a different
    # model's answer, which check_committee cannot detect because it compares sorted
    # names. A typo in the path must still not throw the prediction away, so an
    # unreadable table falls back to the registry order and reports no confidence.
    try:
        table = core.load_calibration(calibration)
    except Exception as exc:
        table = None
        unreadable = f"calibration_unreadable: {type(exc).__name__}"
    else:
        unreadable = None

    priority = core.tie_break_order(table) if table else list(_specialists())
    v = core.vote(results, priority=priority)
    if v["winner"] is None:
        return core.build_result(
            ok=False, backend_used="ensemble", cold_start=cold, latency_s=total,
            error="every specialist failed or returned an unparseable SMILES",
            abstained=v["abstained"],
            confidence_unavailable_reason="no_prediction",
        )

    if unreadable:
        mismatch = None
        conf = {"p": None, "label": "unavailable", "reason": unreadable}
    else:
        mismatch = core.check_committee(v, table)
        if mismatch:
            # Do not silently drop the confidence: the caller asked for the ensemble
            # precisely to get a number, and a partial install is invisible otherwise.
            conf = {"p": None, "label": "unavailable", "reason": mismatch}
        else:
            conf = core.confidence(v["pattern"], table)

    narrow = {"ok": True, "smiles": v["winner"], "raw": "",
              "model_used": "+".join(v["voters"]), "cold_start": cold,
              "latency_s": round(total, 3), "error": ""}
    out = _finish(narrow, agreement=v["pattern"],
                  basis="agreement" if conf.get("p") is not None else None,
                  backend_used="ensemble", conf=conf,
                  votes=v["votes"], abstained=v["abstained"])
    if not out["warning"]:
        # Surface both in warning, not only in the reason: an unreadable path looked
        # identical to a legitimate no-number result in anything that prints the
        # result, and the reason alone named a Python exception class.
        out["warning"] = mismatch or (
            f"the calibration table at {calibration!r} could not be read, so this "
            f"answer carries no confidence" if unreadable else ""
        )
    return out


def image_to_smiles_core(image_path: str, backend: str = "auto",
                         model: str | None = None,
                         calibration: str | None = None,
                         models_wanted: list[str] | None = None,
                         report_solo_accuracy: bool = False) -> dict:
    """Read a molecule's 2D structure diagram and return its SMILES.

    The plain-Python entry point. ``image_to_smiles`` is the agent-facing wrapper.
    Never raises: every failure comes back as ``ok=False`` with an actionable ``error``.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, GIF or WEBP of one molecule's structure.
    backend : str
        See :data:`chemgraph.tools.ocsr_models.BACKENDS`.
    model : str, optional
        Which model, for ``alcf`` and ``shim``. Ignored for the specialists, whose
        backend name is the model.
    calibration : str, optional
        Path to a calibration table, overriding the packaged default.
    models_wanted : list[str], optional
        For ``backend="ensemble"``, run only these specialists. The committee is what
        a calibration table describes, so a table fit on two models yields a number
        only when exactly those two vote.
    report_solo_accuracy : bool, optional
        For a single-specialist backend, return that model's overall benchmark
        accuracy as ``confidence``. Off by default: it is a property of the model on
        someone else's images and says nothing about this one. Only the ensemble
        produces a per-image number.

    Returns
    -------
    dict
        The 18-key contract; see :func:`chemgraph.tools.ocsr_core.build_result`.
    """
    # Check the backend name before touching the filesystem, so a typo is reported as
    # a typo rather than as whatever happens to be wrong with the path. Guarded for
    # hashability first: a list or dict here raised TypeError straight out of the
    # function, past the catch-all below, breaking the never-raise promise.
    if not isinstance(backend, str) or backend not in models.BACKENDS:
        return core.build_result(
            ok=False, backend_used=backend,
            error=(f"unknown backend {backend!r}. Choose one of: "
                   f"{', '.join(models.BACKENDS)}"),
        )

    try:
        image_bytes, mime = core.load_image_bytes(image_path)
    except (OSError, ValueError, TypeError) as e:
        # OSError, not just FileNotFoundError: an agent-synthesized path routinely
        # produces PermissionError, NotADirectoryError ("some/file.txt/x.png") or
        # ENAMETOOLONG, and the docstring promises this returns rather than raises.
        # TypeError covers image_path=None from a direct Python call.
        return core.build_result(ok=False, error=str(e), backend_used=backend)

    try:
        if backend == "auto":
            installed = backends.available_specialists()
            if _default_single() in installed:
                backend = _default_single()
            elif installed:
                # Some other specialist is installed. Saying "no local models are
                # installed" here was false, and the tool docstring tells an agent to
                # answer that by going to a vision LLM, so a compliant agent
                # abandoned a working local model that is more accurate.
                backend = next(
                    name for name in _specialists() if name in installed
                )
                logger.info(
                    "backend='auto': %s is not installed, using %s",
                    _default_single(), backend,
                )
            else:
                return core.build_result(
                    ok=False, backend_used="none",
                    error=(f"{backends._setup_hint()} (backend='auto' uses "
                           f"{_default_single()})"),
                    confidence_unavailable_reason="no_specialists_installed",
                )

        if backend == "ensemble":
            return _run_ensemble(image_bytes, calibration, models_wanted)

        if backend in _specialists():
            narrow = backends.smiles_from_specialist(backend, image_bytes)
            # Pass the table through. Without it prior_confidence loads the packaged
            # default, so a user who refit on their own images got their numbers from
            # the ensemble and someone else's from every single-model backend, which
            # includes the 'auto' default, with nothing in the result to show it.
            conf = (_prior_with(backend, calibration) if report_solo_accuracy
                    else {"p": None, "label": "unavailable",
                          "reason": "single_model_has_no_per_image_confidence"})
            return _finish(narrow, agreement="single",
                           basis="prior" if report_solo_accuracy else None,
                           backend_used=backend, conf=conf)

        if backend in _LLM_BACKENDS:
            narrow = backends.smiles_from_llm(image_bytes, mime, backend, model)
            used = narrow["model_used"]
            resolved = backend
            if backend == "llm":
                # report which endpoint actually answered, not the alias
                resolved = "shim" if used in models.SHIM_VISION_MODELS.values() else "alcf"
            return _finish(narrow, agreement="single", basis=None,
                           backend_used=resolved)

        # Unreachable: BACKENDS was checked above. Kept so a new entry added to
        # BACKENDS without a branch here fails loudly instead of silently.
        return core.build_result(
            ok=False, backend_used=backend,
            error=f"backend {backend!r} is listed but not implemented",
        )
    except Exception as e:  # the never-raise invariant, belt and braces
        logger.exception("unexpected OCSR failure")
        return core.build_result(ok=False, backend_used=backend,
                                 error=f"{type(e).__name__}: {e}")


@tool
def image_to_smiles(image_path: str, backend: str = "auto",
                    model: str | None = None) -> dict:
    """Read a molecule's 2D structure diagram from an image and return its SMILES.

    Use this when the user supplies an image file of a chemical structure.

    Backends. Use 'auto' unless you have a reason not to; it is the default and is
    right for almost every question. Switch only in these cases:
      - the answer will drive an expensive calculation (a geometry optimization, a
        DFT job): use 'ensemble', which runs every installed specialist and returns
        a sharper confidence, at the cost of being slower.
      - 'auto' returned ok=False saying no local models are installed: use 'alcf'.
      - the user named a specific model: pass its name as the backend, or use
        backend='alcf'/'shim' with model=. The error message on an unknown backend
        lists every name this installation accepts; do not guess from memory.

    Do NOT try backends in a loop. If one fails because nothing is installed, the
    others in that family fail the same way, and each attempt can cost 45 seconds.
    The error says which family works on this machine; believe it.

    ALWAYS read `confidence` before acting on the answer:

      - backend='ensemble' returns a number, and `basis` is 'agreement'. It is how
        often this exact agreement pattern gave the right answer on measured data.
        Above about 0.95 the answer is safe to feed to an expensive calculation;
        below it, say so to the user or ask them to check the structure.
      - Every other backend returns confidence=null with a reason. A single model
        cannot say how likely IT is to be right about THIS image, so nothing is
        quoted. Null means unmeasured here, and never that the answer is bad.
      - If the answer matters, use backend='ensemble', the only one that produces a
        per-image number.

    `confidence_label` is the coarse version and is the safer thing to reason with,
    because a bare number can look more precise than it is. It is 'unanimous',
    'strong', 'weak' or 'conflicting' when a number was produced; the same four
    prefixed 'low_n_' when the bucket is too small to quote one, which still tells
    you which way the evidence points; and 'unavailable' when no measurement applies
    at all.

    A null `confidence` means the confidence machinery failed, not that the answer is
    good; `confidence_unavailable_reason` says why.

    If `n_fragments` is greater than 1 the image contained more than one molecule (or
    a salt, or a reaction scheme). Ask the user which one they meant rather than
    passing the SMILES to a geometry or energy calculation.

    Timing. The first call in a session loads models: about 55-65 s for DECIMER and
    5-20 s for the others, so a cold 'ensemble' can take well over a minute of wall
    time. Later calls take about 0.7 s ('auto') or 3-5 s ('ensemble'). `cold_start`
    says whether a load happened; `latency_s` counts inference only and excludes that
    load, so on a cold call the wall time is much larger than the number reported.

    Specialists are purpose-built and usually more accurate than a general vision
    model, but fail on unusual drawing styles, Markush structures, and reaction
    schemes.

    Parameters
    ----------
    image_path : str
        Path to a PNG or JPEG showing ONE molecule's 2D structure.
    backend : str, optional
        'auto', 'ensemble', a specialist name, 'alcf', 'shim', or 'llm'.
    model : str, optional
        Which vision model, for 'alcf' and 'shim'.

    Returns
    -------
    dict
        Always present: ok, smiles, valid, formula, n_fragments, confidence,
        confidence_label, confidence_unavailable_reason, agreement, basis,
        backend_used, model_used, cold_start, latency_s, error, warning.
        Only for backend='ensemble': votes, abstained.
    """
    return image_to_smiles_core(image_path, backend=backend, model=model)


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
