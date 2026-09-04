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

# Cap on stereoisomer enumeration when comparing two readings of one skeleton. 512
# covers every prediction in the benchmark (max 512, 8 of 2777 above 64), and the
# count is checked first so a molecule past it takes the substructure path instead
# of comparing a truncated set.
_MAX_ISOMERS = 512


def measured_accuracies() -> dict[str, dict]:
    """Solo accuracy per model, from the calibration table rather than the registry.

    The registry describes what a model is: how to run it, how fast, what it needs.
    How well it did is a measurement, and it belongs with the data that produced it,
    so a user who refits on their own images sees their own numbers here. An
    unreadable table costs the accuracies and not the listing.
    """
    try:
        table = core.load_calibration()
    except (OSError, ValueError, TypeError):
        return {}
    out = {}
    for m in models.SPECIALIST_MODELS:
        entry = core.model_performance(m, table)
        # The listing shows the number the tool quotes, which is the table's 'p'.
        # 'accuracy' beside it is the raw rate, the record of what was counted, and
        # showing that here would have one install report two different accuracies
        # for one model. Withheld below the floor, so the listing cannot state a
        # figure the tool refuses to state.
        if entry:
            prior = core.prior_confidence(m, table)
            entry["accuracy"] = prior["p"] if prior["reason"] is None else None
        out[m] = entry
    return out


def _unknown_model_error(model: str) -> str:
    installed = backends.available_specialists()
    return (f"unknown model {core.echo(repr(model))}. Choose one of: "
            f"{', '.join(models.MODEL_CHOICES)}.\n\n"
            f"{models.describe_models(installed, measured_accuracies(), backends.usable_specialists())}")


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

    if not isinstance(model, str):
        # models_wanted is type-checked and this was not, so a non-string reached
        # .strip() and raised AttributeError past the never-raises contract.
        return "", (f"model must be a name, got {type(model).__name__}. "
                    f"Choose one of: {', '.join(models.MODEL_CHOICES)}.")
    # Through str's own methods: isinstance passes for a subclass, which is free to
    # override strip and lower with anything, including a raise. No str() call
    # either, since __str__ is equally overridable and isinstance already passed.
    name = str.lower(str.strip(model))
    if name not in models.MODEL_CHOICES:
        return "", _unknown_model_error(model)
    return name, ""


def _validate_models_wanted(wanted, installed: list[str]) -> str:
    """Return an error string for an unusable models_wanted, or "" when it is fine.

    A bare string iterates per character, and a non-string element blows up the join
    below with a TypeError naming neither the argument nor the valid names. An empty
    list is a caller who filtered down to nothing, which must not silently become
    "run everything".
    """
    choices = ", ".join(models.SPECIALIST_MODELS)
    if isinstance(wanted, str) or not isinstance(wanted, (list, tuple, set, frozenset)):
        return (f"models_wanted must be a list of specialist names, got "
                f"{type(wanted).__name__}. Choose from: {choices}")
    if not wanted:
        return (f"models_wanted is empty. Omit it to vote every installed "
                f"specialist, or name some of: {choices}")
    # isinstance first: `m not in dict` hashes m, so an unhashable element raises
    # from inside the guard that exists to report it.
    unknown = [m for m in wanted
               if not isinstance(m, str) or m not in models.SPECIALIST_MODELS]
    if unknown:
        # Named by type where the value is not a string: repr is caller-supplied
        # code too, and this is the guard that has to survive whatever it is given.
        shown = ", ".join(core.echo(repr(m)) if isinstance(m, str)
                          else f"a {type(m).__name__}" for m in unknown)
        return f"not OCSR specialists: {shown}. Choose from: {choices}"
    absent = [m for m in wanted if m not in installed]  # all strings by now
    if absent:
        return (f"requested but not installed: {', '.join(absent)}. Install with: "
                f"pip install 'chemgraph[ocsr]'")
    return ""


def _prior_label(model: str, calibration: str | None) -> str:
    """Band a model's measured accuracy, from the table the caller named.

    Without threading ``calibration`` through, a user who refit on their own images
    and passed the path would get their own numbers from a committee and the
    packaged ones from every single-model read, with nothing in the result to show
    the two came from different data.
    """
    try:
        table = core.load_calibration(calibration)
    except (OSError, ValueError, TypeError):
        return "unavailable"
    return core.prior_confidence(model, table)["label"]


def _run_ensemble(resolved: str, calibration: str | None,
                  models_wanted: list[str] | None) -> dict:
    """Vote a committee of specialists and attach a calibrated confidence.

    Runs every installed specialist unless ``models_wanted`` names a subset. A subset
    is worth supporting because the committee is what a table describes: someone who
    has all four installed but fitted a table on two of them can only get a number by
    running exactly those two.
    """
    installed = backends.available_specialists()
    if not installed:
        return core.build_result(
            ok=False, backend_used="none", error=backends._install_hint(),
            confidence_unavailable_reason="no_specialists_installed",
        )

    if models_wanted is not None:
        error = _validate_models_wanted(models_wanted, installed)
        if error:
            return core.build_result(ok=False, backend_used="ensemble", error=error)
        installed = [m for m in installed if m in models_wanted]

    results, absent, cold, total = [], [], False, 0.0
    for name in installed:
        r = backends.smiles_from_specialist(name, resolved)
        cold = cold or r["cold_start"]
        total += r["latency_s"]
        if not r.get("ran", True):
            # It never saw the image: no checkpoint, or the load failed. Voting it
            # as a dissenting singleton would read as disagreement about the
            # picture, and a table fit on four working models would score three
            # such entries at the all-different rate. Drop it from the committee
            # and let check_committee report the smaller set.
            absent.append((name, r["error"]))
            continue
        results.append({"model": name, "smiles": r["smiles"], "ok": r["ok"],
                        "error": r["error"]})

    if not results:
        # A separate reason from no_specialists_installed: the extra is installed
        # here and the remedy is whatever each error names, usually a checkpoint to
        # download. Telling this caller to pip install would send them to a no-op.
        return core.build_result(
            ok=False, backend_used="ensemble", cold_start=cold,
            latency_s=round(total, 3),
            error="no specialist could run: " + core.echo(
                "; ".join(f"{n}: {e}" for n, e in absent), 400),
            confidence_unavailable_reason="no_specialist_could_run",
        )

    # Load before voting: the table records the model priority it was fitted under,
    # and that order decides which answer wins a tie. Voting by the registry's order
    # would attach a number measured for one model's answer to a different model's
    # answer, which check_committee cannot detect because it compares sorted names.
    # A typo in the path must still not throw the prediction away, so an unreadable
    # table falls back to the registry order and reports no confidence.
    try:
        table = core.load_calibration(calibration)
    except (OSError, ValueError, TypeError) as exc:
        table, unreadable = None, f"calibration_unreadable: {type(exc).__name__}"
    else:
        unreadable = None

    priority = core.tie_break_order(table) if table else list(models.SPECIALIST_MODELS)
    v = core.vote(results, priority=priority)
    if v["winner"] is None:
        return core.build_result(
            ok=False, backend_used="ensemble", cold_start=cold,
            latency_s=round(total, 3), abstained=v["abstained"],
            error="every specialist failed or returned an unparseable SMILES",
            confidence_unavailable_reason="no_prediction",
        )

    mismatch = None if table is None else core.check_committee(v, table)
    if unreadable:
        conf = {"p": None, "label": "unavailable", "reason": unreadable}
    elif mismatch:
        # Do not silently drop the confidence: the caller asked for the ensemble
        # precisely to get a number, and a partial install is invisible otherwise.
        conf = {"p": None, "label": "unavailable", "reason": mismatch}
    else:
        conf = core.confidence(v["pattern"], table)

    # vote() groups stereo-blind, because that is how the table was fit, so its
    # winner is the stereo-stripped key. Return a form that keeps the wedge bonds a
    # model resolved: a unanimous committee must not answer with a racemate where
    # the single-model path answers with one enantiomer. The strongest model in the
    # winning group decides, by the same priority that breaks ties, since members
    # can disagree on stereochemistry while agreeing on the skeleton.
    warnings: list[str] = []
    winners = v["votes"][v["winner"]]
    # vote() stores bare names, so the priority has to be bare too. A table writing
    # "local:molnextr" would otherwise match nothing and fall through to insertion
    # order, which is the arbitrary choice this whole block exists to avoid.
    order = [n.removeprefix("local:") for n in priority]
    stereo_votes: dict[str, list[str]] = {}
    for r in results:
        model = r["model"].removeprefix("local:")
        if model not in winners:
            continue
        form = core.canonicalize(r["smiles"], stereo=True)
        if form is not None:
            stereo_votes.setdefault(form, []).append(model)

    if stereo_votes:
        # The strongest model in the group supplies the stereochemistry, by the same
        # priority that breaks ties. Counting the group again and taking the majority
        # was measured on the benchmark's 47 stereo-bearing items and rejected: the
        # models rank the same way on stereochemistry as they do overall (DECIMER
        # 76.6%, molnextr 57.4%, ocsrglyph 48.9%, molscribe 40.4%), so a majority
        # lets three weaker readings outvote the one most likely to be right.
        smiles = next((f for m in order for f in stereo_votes if m in stereo_votes[f]),
                      next(iter(stereo_votes)))

        # Warn when the answer loses something another member read. A member that
        # marked less is not a conflict: reading one of two double bonds where the
        # answer reads both discards nothing. What matters is whether the answer
        # still carries every centre that member assigned, which RDKit answers by
        # comparing what each form still leaves open. On the benchmark 115 of 721
        # groups hold more than one form; 112 lose something and warn, and in the
        # other 3 the answer already says everything the rest did.
        def _covered_by_answer(other: str) -> bool:
            """True when the answer says everything `other` says, and agrees.

            Compared as sets of the stereoisomers each form still admits: the
            answer covers the other exactly when it leaves no more of them open.
            That needs no alignment between the two molecules, which is what makes
            it right where comparing perceived stereo elements is not. Those come
            back in an order RDKit does not promise, in a count that changes when
            assigning one centre makes another non-stereogenic, and carrying a
            descriptor of NoValue for every class except tetrahedral, so square
            planar sulfur and octahedral metals compare equal whatever they say.
            """
            from rdkit import Chem
            from rdkit.Chem.EnumerateStereoisomers import (
                EnumerateStereoisomers, GetStereoisomerCount,
                StereoEnumerationOptions)

            mol = Chem.MolFromSmiles(smiles)
            ref = Chem.MolFromSmiles(other)
            if mol is None or ref is None:
                return False
            opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True,
                                            tryEmbedding=False, maxIsomers=_MAX_ISOMERS)
            try:
                # Count first. Past the cap EnumerateStereoisomers returns exactly
                # _MAX_ISOMERS forms with nothing to say it stopped early, and a
                # truncated set is not a subset of anything: a sugar read with every
                # centre assigned would report a conflict against the same sugar read
                # with none, which is the case this function exists to allow. Fall
                # back to asking whether the answer embeds the other with its
                # chirality intact. The count overestimates, since it honours
                # neither the cap nor unique=True, so this path is taken by some
                # molecules the enumeration could have finished; the two rules agree
                # on all 115 multi-form groups in the benchmark.
                if (GetStereoisomerCount(mol, options=opts) > _MAX_ISOMERS
                        or GetStereoisomerCount(ref, options=opts) > _MAX_ISOMERS):
                    # Blind to square planar and octahedral centres, which
                    # HasSubstructMatch ignores. Comparing their tags was tried and
                    # reverted: _chiralPermutation indexes the neighbour order the
                    # SMILES was written in, so the same geometry reached two ways
                    # compares unequal and 12 of 18 same-geometry pairs were
                    # reported as conflicts. A rare missed conflict beats frequent
                    # false ones on the case this function exists to allow.
                    return (mol.GetNumAtoms() == ref.GetNumAtoms()
                            and mol.HasSubstructMatch(ref, useChirality=True))
                mine = {Chem.MolToSmiles(x)
                        for x in EnumerateStereoisomers(mol, options=opts)}
                theirs = {Chem.MolToSmiles(x)
                          for x in EnumerateStereoisomers(ref, options=opts)}
            except Exception:  # pragma: no cover - defensive around RDKit
                return False
            # No emptiness guard: a molecule RDKit parsed always enumerates to at
            # least itself, and the one case that produced a misleading set, the
            # cap cutting the enumeration short, is caught above by the count.
            return mine <= theirs

        conflicting = [f for f in stereo_votes
                       if f != smiles and not _covered_by_answer(f)]
        if conflicting:
            # Name the support for the answer, not the largest group. The two are
            # often different: the strongest model is regularly the one that read
            # no stereocentre where the rest did, so a leading count would read as
            # the answer's backing when it belongs to the readings it overruled.
            backing = sorted(stereo_votes[smiles])
            outvoted = sum(len(m) for f, m in stereo_votes.items() if f != smiles)
            warnings.append(
                f"the committee agreed on the skeleton and read "
                f"{len(stereo_votes)} different stereochemistries; this answer is "
                f"{', '.join(backing)}'s reading and {outvoted} other "
                f"{'model' if outvoted == 1 else 'models'} read otherwise. The "
                f"confidence was measured stereo-blind and does not cover it")
    else:
        smiles = v["winner"]

    validation = core.validate_smiles_core(smiles)
    # Both can be true at once: a salt read by a committee whose table is missing
    # needs both caveats, so they accumulate rather than shadowing each other.
    fragments = core.fragment_warning(validation)
    if fragments:
        warnings.insert(0, fragments)
    if absent:
        # Independent of everything below: a committee can shrink and still find a
        # table that fits the survivors, which reports a confidence and no mismatch.
        # Nested under one of those branches, the only sign that three of four
        # models never ran would be a latency nobody reads.
        warnings.append("These were installed but could not run: " + core.echo(
            "; ".join(f"{n}: {e}" for n, e in absent), 400))
    if mismatch:
        # Surface this in warning too: the reason alone names a Python exception
        # class, and anything that prints the result shows a bare missing number.
        warnings.append(mismatch)
    elif unreadable:
        warnings.append(f"the calibration table at {core.echo(repr(calibration))} "
                        f"could not be read, so this answer carries no confidence")
    elif conf.get("reason") == "unknown_pattern":
        # The third confidence-less path. Without this it is the only one that
        # surfaces as a bare missing number, which is what the other two get a
        # warning to prevent.
        warnings.append(f"the calibration table has no {v['pattern']!r} bucket, so "
                        f"this split carries no measured confidence")
    warning = " ".join(warnings)

    return core.build_result(
        ok=True,
        smiles=smiles,
        valid=validation.get("valid", False),
        formula=validation.get("formula"),
        n_fragments=validation.get("n_fragments", 0),
        confidence=conf["p"],
        # Carried even where p is withheld: below the sample floor the interval is
        # the only quantitative thing a thin bucket can honestly offer.
        confidence_interval=conf.get("ci"),
        confidence_label=conf["label"],
        confidence_unavailable_reason=conf.get("reason"),
        agreement=v["pattern"],
        backend_used="ensemble",
        model_used="+".join(v["voters"]),
        cold_start=cold,
        latency_s=round(total, 3),
        warning=warning,
        votes=v["votes"],
        abstained=v["abstained"],
    )


def image_to_smiles_core(image_path: str, model: str | None = None,
                         structured: bool = False, ensemble: bool = False,
                         calibration: str | None = None,
                         models_wanted: list[str] | None = None,
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
    ensemble : bool, optional
        Run every installed specialist and vote, attaching a measured confidence.
        Costs one inference per model. ``model`` is ignored when this is set.
    calibration : str, optional
        Path to a calibration table. Defaults to ``CHEMGRAPH_OCSR_CALIBRATION``,
        then the packaged four-model table.
    models_wanted : list of str, optional
        With ``ensemble``, vote only these specialists instead of every installed
        one. Needed when the table describes a subset of what is installed.
    structured : bool, optional
        With ``model="llm"``, ask for a JSON reply. Ignored by the specialists,
        which take no prompt.
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
    if not isinstance(image_path, str):
        return core.build_result(
            error=f"image_path must be a path, got {type(image_path).__name__}")

    try:
        resolved = core.resolve_image_path(image_path)
        image_bytes, mime = core.load_image_bytes(resolved)
    except (FileNotFoundError, ValueError) as e:
        return core.build_result(model_used=name, error=str(e))
    except OSError as e:
        # Both halves bounded: an OSError's own text repeats the path it failed on,
        # so quoting the exception unbounded reintroduces what echo just trimmed.
        return core.build_result(
            model_used=name,
            error=f"cannot read {core.echo(image_path)}: {core.echo(e, 200)}")

    if ensemble:
        # Dispatched after the image checks above, so a committee run rejects a bad
        # file once instead of once per model.
        return _run_ensemble(resolved, calibration, models_wanted)

    if name == models.LLM_MODEL:
        narrow = backends.smiles_from_llm(image_bytes, mime, llm,
                                          structured=structured)
    else:
        # The resolved path, so a specialist opens the same file that was validated.
        narrow = backends.smiles_from_specialist(name, resolved)

    if not narrow["ok"]:
        return core.build_result(
            model_used=narrow.get("model_used") or name,
            cold_start=narrow.get("cold_start", False),
            latency_s=narrow.get("latency_s", 0.0),
            error=narrow.get("error") or "the model returned no SMILES",
        )

    validation = core.validate_smiles_core(narrow["smiles"])

    # Canonical form, so two models that read the same structure return the same
    # string: DECIMER writes Kekule SMILES where the others write aromatic. Keeps
    # stereochemistry, unlike validation's `canonical_smiles`, which drops it to match
    # how the benchmark scores. Falls back to the raw string when RDKit could not
    # parse it, since an unparseable answer is still worth reporting with valid=False.
    smiles = core.canonicalize(narrow["smiles"], stereo=True) or narrow["smiles"]

    warning = core.fragment_warning(validation)

    return core.build_result(
        ok=True,
        smiles=smiles,
        valid=validation.get("valid", False),
        formula=validation.get("formula"),
        n_fragments=validation.get("n_fragments", 0),
        # One model produced one answer, so there is no agreement to score, and
        # confidence stays None. Naming the reason keeps this apart from a committee
        # whose table failed to load, which also reports no number. The label still
        # carries the model's measured solo accuracy, which is the question a caller
        # asks next and the only one a single read can answer.
        confidence_label=_prior_label(narrow.get("model_used") or name, calibration),
        confidence_unavailable_reason="single_model_has_no_per_image_confidence",
        backend_used="llm" if name == models.LLM_MODEL else "specialist",
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

    Do NOT loop through every model hoping one succeeds. A cold model costs 9-170 s to
    load, and if the image is unreadable they usually all fail the same way. Two
    attempts is a reasonable ceiling.

    Check `valid` and `n_fragments` before acting on the answer. `valid` is false when
    RDKit could not parse what the model produced. `n_fragments` above 1 means the
    image held more than one molecule (a salt, a mixture, a reaction scheme): ask the
    user which one they meant instead of passing the SMILES to a geometry or energy
    calculation.

    A single-model call reports no confidence: one model cannot say how likely it is
    to be right about this particular image, and its benchmark accuracy describes
    someone else's images. `ensemble=True` measures it instead, by reading the image
    with every installed specialist and looking up how often a committee splitting
    that way was right.

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
    structured : bool, optional
        With model='llm', ask for a JSON reply. Leave it off unless the model is
        returning prose the tool cannot read. No effect on the specialists.
    ensemble : bool, optional
        Read the image with every installed specialist and vote. Returns a measured
        confidence in `confidence`, which a single model cannot give. Costs one
        inference per model, so use it when the answer matters more than the time:
        before a calculation, or after a plain call returned something doubtful.
        Ignores `model`.
    models_wanted : list of str, optional
        With ensemble, vote only these specialists. Omit it to vote all installed.

    Returns
    -------
    dict
        ok : bool
            Whether a parseable SMILES was produced.
        smiles : str or None
            The SMILES, canonicalized by RDKit with stereochemistry kept, so two
            models that read the same structure return the same string.
        valid : bool
            Whether RDKit parsed it.
        formula : str or None
            Molecular formula, when valid.
        n_fragments : int
            Disconnected components; above 1 means more than one molecule.
        confidence : float or None
            P(this answer is correct), when a committee measured it. None from a
            single model, which has no agreement to score.
        confidence_interval : list or None
            The 95% interval behind that number. Present even where `confidence`
            is withheld for a thin bucket, which is the case it matters most in.
        confidence_label : str
            One of 'unanimous' (p >= 0.99), 'strong' (>= 0.95), 'weak' (>= 0.70),
            or 'conflicting' below that, prefixed 'low_n_' when the bucket is too
            thin to quote a number. 'unknown' means the table has no bucket for
            this split, and 'unavailable' that no number applies at all. On a
            single-model call it bands that model's measured solo accuracy, since
            there is no per-image number to band.
        confidence_unavailable_reason : str or None
            Why there is no number, when there is none.
        agreement : str or None
            How a committee split, as 'majority/rest', e.g. '4' or '3/1'.
        votes : dict or None
            Which models produced which SMILES.
        abstained : dict or None
            Models that ran and returned nothing usable.
        backend_used : str or None
            'specialist', 'llm', or 'ensemble'.
        model_used : str
            Which model answered.
        cold_start : bool
            Whether this call paid to load the model.
        latency_s : float
            Seconds, including the load when cold_start is true.
        error : str
            Empty when ok, otherwise what went wrong and what to do about it.
        warning : str
            Non-fatal caveats: multiple fragments, a committee the table does not
            describe, models that could not run, and a committee that agreed on
            the skeleton while reading different stereochemistry. The last one can
            accompany a high confidence, because the table was fitted stereo-blind
            and so scores the skeleton alone.
    """


def make_ocsr_tools(llm: Any = None) -> list:
    """Build the OCSR tools, binding ``llm`` as the vision fallback.

    The agent passes its own chat model here, which is what lets ``model='llm'`` work
    without this module resolving any credentials. Called with no model the fallback
    reports that it is unavailable instead of failing at call time inside an API
    client.
    """

    def _run(image_path: str, model: str | None = None,
             structured: bool = False, ensemble: bool = False,
             models_wanted: list[str] | None = None) -> dict:
        return image_to_smiles_core(image_path, model=model, structured=structured,
                                    ensemble=ensemble, models_wanted=models_wanted,
                                    llm=llm)

    _run.__doc__ = _TOOL_DOC
    return [StructuredTool.from_function(
        func=_run,
        name="image_to_smiles",
        description=_TOOL_DOC,
    )]


@tool
def image_to_smiles(image_path: str, model: str | None = None,
                    ensemble: bool = False,
                    models_wanted: list[str] | None = None) -> dict:
    """Read a molecule's 2D structure diagram from an image and return its SMILES.

    Module-level tool for callers that bind tools statically. It has no LLM bound, so
    `model='llm'` reports the fallback as unavailable; use `make_ocsr_tools(llm)` to
    get a version with the agent's model wired in. Every other model works here,
    committee voting included: the committee is specialists only.

    See `make_ocsr_tools` for the full contract.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, GIF or WEBP showing ONE molecule's 2D structure.
    model : str, optional
        'decimer', 'molnextr', 'molscribe', 'ocsrglyph', or 'llm'.
    ensemble : bool, optional
        Vote every installed specialist and return a measured confidence.
    models_wanted : list of str, optional
        With ensemble, vote only these specialists. Omit it to vote all installed.
    """
    return image_to_smiles_core(image_path, model=model, ensemble=ensemble,
                                models_wanted=models_wanted, llm=None)


@tool
def list_ocsr_models() -> str:
    """List the OCSR models, their measured accuracy, and which are installed here.

    Use this when the user asks what models are available, or after an install error,
    so the answer names what this machine actually has instead of guessing.
    """
    return models.describe_models(backends.available_specialists(),
                                  measured_accuracies(),
                                  backends.usable_specialists())


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
