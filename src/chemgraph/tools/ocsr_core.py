"""Pure-Python OCSR helpers (no LangChain / MCP decorators).

Optical Chemical Structure Recognition: read a molecule's 2D structure diagram
from an image and return its SMILES. Used by the LangChain ``@tool`` wrappers in
:mod:`chemgraph.tools.ocsr_tools`.

Two families of backend. A vision LLM reads the picture directly; local specialist
models (MolNexTR, MolScribe, DECIMER, OCSRGlyph) are purpose-built image-to-SMILES
networks driven as subprocesses. Running several specialists and looking at how much
they agree yields a calibrated confidence, which is the point of :func:`vote` and
:func:`confidence`.

RDKit is a core dependency and is imported lazily anyway, so this module imports and
its tests collect on a host with no specialist environment and no network. A
:func:`mock_ocsr` helper provides deterministic output for hermetic tests.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import stat

logger = logging.getLogger(__name__)

# Reject anything that is not one of these before base64-ing it to a remote endpoint.
# Sniffed from the leading bytes, never from the file extension: the extension is
# attacker-controlled and the whole point is to refuse a non-image renamed to .png.
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}
_MAX_IMAGE_BYTES = 8_000_000

# Confidence label cut-points, applied to the point estimate. Below the calibration
# table's n floor these are prefixed "low_n_", because a 30-60 pp interval does not
# deserve a decimal.
_LABEL_BANDS = ((0.99, "unanimous"), (0.95, "strong"), (0.70, "weak"))

# RDKit parsing is superlinear in string length: 20k characters costs about 9 s of
# CPU and a megabyte-scale string is effectively unbounded. Model output is untrusted,
# so cap it before RDKit sees it. Real SMILES are well under 1000 characters; the
# longest in the OCSR benchmark is 224.
_MAX_SMILES_CHARS = 4000

_SMILES_LABEL_RE = re.compile(r"^\s*(?:smiles|answer|result)\s*[:=]\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _sniff_mime(head: bytes) -> str | None:
    """Return the MIME type implied by a file's leading bytes, or None.

    WEBP needs a split check: "RIFF" then "WEBP" four bytes later.
    """
    for magic, mime in _MAGIC.items():
        if head.startswith(magic):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def load_image_bytes(image_path: str, max_bytes: int = _MAX_IMAGE_BYTES) -> tuple[bytes, str]:
    """Read an image file safely and return ``(raw_bytes, mime_type)``.

    Validates before reading rather than after. Reading first and checking later is
    not equivalent: ``/dev/zero`` grows the buffer without bound, a FIFO blocks
    forever in ``open()`` so no caller-side timeout fires, and a 50 GB regular file
    is fully resident before any size check could reject it.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, GIF or WEBP file.
    max_bytes : int, optional
        Reject anything larger, by default 8 MB.

    Returns
    -------
    tuple[bytes, str]
        The file contents and its sniffed MIME type.

    Raises
    ------
    FileNotFoundError
        No such path.
    ValueError
        Not a regular file, too large, or not a recognized image format.
    """
    path = os.path.abspath(os.path.expanduser(image_path))
    try:
        st = os.lstat(path)  # lstat, so a symlink is judged on its own terms
    except FileNotFoundError:
        raise FileNotFoundError(f"no such image: {path}")

    if not stat.S_ISREG(st.st_mode):
        raise ValueError(
            f"not a regular file: {path}. Directories, FIFOs, sockets and device "
            f"files are refused (a FIFO would block forever, /dev/zero would not end)."
        )
    if st.st_size > max_bytes:
        raise ValueError(f"image is {st.st_size} bytes, over the {max_bytes} limit: {path}")

    with open(path, "rb") as fh:
        data = fh.read(max_bytes + 1)  # +1 so a file that grew since lstat is caught
    if len(data) > max_bytes:
        raise ValueError(f"image grew past the {max_bytes} limit while reading: {path}")

    mime = _sniff_mime(data[:16])
    if mime is None:
        raise ValueError(
            f"not a recognized image (PNG/JPEG/GIF/WEBP) by content: {path}. "
            f"The extension is not trusted; only the leading bytes are."
        )
    return data, mime


def load_image_b64(image_path: str, max_bytes: int = _MAX_IMAGE_BYTES) -> tuple[str, str]:
    """Read an image and return ``(base64_ascii, mime_type)`` for a data URL."""
    data, mime = load_image_bytes(image_path, max_bytes=max_bytes)
    return base64.b64encode(data).decode("ascii"), mime


def extract_image_path(text: str) -> str | None:
    """Pull the first usable image path out of a free-text query.

    A candidate must both exist and pass the magic-byte check, so a text file named
    ``notes.png`` is never selected.

    Parameters
    ----------
    text : str
        Natural-language query, e.g. "what is the SMILES in /data/mol.png?".

    Returns
    -------
    str or None
        Absolute path to the first real image mentioned, else None.
    """
    if not text:
        return None
    # The optional drive prefix keeps a Windows path whole: without it the match
    # starts after "C:", and joining the remainder against the current drive points
    # somewhere else entirely.
    pattern = r"(?:[A-Za-z]:)?[\w./~\\-]+\.(?:png|jpe?g|gif|webp|bmp|tiff?)"
    for token in re.findall(pattern, text, flags=re.IGNORECASE):
        candidate = token.strip("'\"")
        try:
            load_image_bytes(candidate)
        except Exception:
            continue
        return os.path.abspath(os.path.expanduser(candidate))
    return None


# ---------------------------------------------------------------------------
# SMILES handling
# ---------------------------------------------------------------------------


def canonicalize(smiles: str | None, stereo: bool = False) -> str | None:
    """Canonical SMILES, or None when the string does not parse.

    Defaults to stereo-blind, matching how the OCSR benchmark scores: most reference
    labels carry no stereochemistry, so comparing with it makes a model that correctly
    reads a wedge bond look wrong.

    Grouping predictions by this, rather than by raw string, is load-bearing for
    :func:`vote`. DECIMER emits Kekule SMILES while the other specialists emit
    aromatic, so raw-string comparison finds unanimity on 12 of 422 benchmark items
    where canonical comparison finds it on 289.
    """
    if not smiles or len(smiles) > _MAX_SMILES_CHARS:
        return None
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _looks_like_prose(s: str) -> bool:
    """True if a string is a sentence rather than a SMILES.

    Needed because RDKit is happy to parse prose. "I cannot process images." yields
    a molecule: ``I`` is iodine and the parser stops at the space. Without this
    guard a model's refusal is returned as a confident one-atom prediction.

    Deliberately a syntactic check, not a model call. Asking an LLM "is this a
    refusal?" would add a network round trip per prediction (four per ensemble call),
    and would itself be a model that can refuse. Refusals and SMILES differ in form,
    not in meaning, so form is enough.

    Measured on real data: 0 false positives over 1916 unique specialist predictions,
    and 100% recall over 2100 real reference SMILES wrapped in seven common LLM
    phrasings (fenced, "SMILES:", "The SMILES string is", and so on).

    Known limit: a refusal that happens to be a single organic-subset word would slip
    through. None exists in practice, since refusals are sentences. If a model is ever
    found to emit one, add it to an explicit denylist rather than reaching for a
    classifier.
    """
    if " " in s.strip():
        return True
    # A SMILES is mostly non-alphanumeric structure or organic-subset letters. A word
    # like "images" is all lowercase letters and no structural characters at all.
    return s.isalpha() and s.islower() and not set(s) <= set("cnopsbrifhe")


def extract_smiles(raw_text: str | None) -> str | None:
    """Pull a SMILES out of a model's raw reply.

    Handles a bare string, a fenced code block, and "The SMILES is: X" prose. Returns
    None for a refusal ("I cannot process images"), which must not be mistaken for a
    prediction. Does not validate; :func:`validate_smiles_core` does that.
    """
    if not raw_text or len(raw_text) > _MAX_SMILES_CHARS:
        return None
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw_text).replace("```", "")
    for line in text.splitlines():
        cleaned = _SMILES_LABEL_RE.sub("", line).strip().strip("`\"'").strip()
        if cleaned and not _looks_like_prose(cleaned) and canonicalize(cleaned) is not None:
            return cleaned
    for token in text.split():
        cleaned = token.strip("`\"'").rstrip(".,;!?")
        if (
            len(cleaned) > 1
            and not _looks_like_prose(cleaned)
            and canonicalize(cleaned) is not None
        ):
            return cleaned
    return None


def validate_smiles_core(smiles: str) -> dict:
    """Validate a SMILES with RDKit and report everything a caller can act on.

    Never raises. An unparseable string yields ``valid=False`` with RDKit's own
    sanitization message in ``errors``, which is the useful part: "Explicit valence
    for atom # 3 N, 4, is greater than permitted" tells a model what to fix.

    ``n_fragments`` above 1 means the string describes several disconnected molecules
    (two structures in one image, a salt, a reaction scheme). That case parses,
    sanitizes and canonicalizes cleanly, so nothing else catches it, and it corrupts
    anything downstream: RDKit has no reason to separate disconnected fragments when
    embedding, so they are placed overlapping (measured: 0.20 A closest approach,
    0.00 A between centroids for ``CCO.CCN``), and an energy evaluation on that
    geometry is meaningless while reporting success.

    Returns
    -------
    dict
        smiles, valid, canonical_smiles, formula, n_atoms, n_heavy_atoms,
        elements, n_fragments, errors, rdkit_available
    """
    out = {
        "smiles": smiles,
        "valid": False,
        "canonical_smiles": None,
        "formula": None,
        "n_atoms": 0,
        "n_heavy_atoms": 0,
        "elements": [],
        "n_fragments": 0,
        "errors": [],
        "rdkit_available": True,
    }
    if not smiles or not isinstance(smiles, str):
        out["errors"].append("empty or non-string SMILES")
        return out
    if len(smiles) > _MAX_SMILES_CHARS:
        out["errors"].append(
            f"SMILES is {len(smiles)} characters, over the {_MAX_SMILES_CHARS} limit; "
            f"parsing cost grows superlinearly and real SMILES are far shorter"
        )
        return out

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors
    except ImportError as e:  # pragma: no cover - rdkit is a core dependency
        # Fail open. Failing closed would drive a validation retry loop until the
        # recursion limit, spending a whole budget to produce nothing.
        logger.warning("RDKit unavailable, skipping SMILES validation: %s", e)
        out.update(rdkit_available=False, valid=True)
        return out

    RDLogger.DisableLog("rdApp.*")
    if ">>" in smiles or ">" in smiles.replace("->", ""):
        out["errors"].append(
            "this looks like a reaction SMILES (contains '>'), not a single molecule"
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Re-parse without sanitizing to recover the specific complaint.
        try:
            raw = Chem.MolFromSmiles(smiles, sanitize=False)
            if raw is not None:
                Chem.SanitizeMol(raw)
        except Exception as e:
            out["errors"].append(str(e))
        if not out["errors"]:
            out["errors"].append("RDKit could not parse this SMILES")
        return out

    mol_h = Chem.AddHs(mol)
    stereo_free = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(stereo_free)
    out.update(
        valid=True,
        canonical_smiles=Chem.MolToSmiles(stereo_free),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        n_atoms=mol_h.GetNumAtoms(),
        n_heavy_atoms=mol.GetNumAtoms(),
        elements=sorted({a.GetSymbol() for a in mol_h.GetAtoms()}),
        n_fragments=len(Chem.GetMolFrags(mol)),
    )
    if out["n_fragments"] > 1:
        out["errors"].append(
            f"this SMILES describes {out['n_fragments']} disconnected molecules; "
            f"ask which one is meant before using it for a calculation"
        )
    return out


def compare_smiles(smiles_a: str, smiles_b: str) -> dict:
    """Report how two SMILES differ: same molecule, same skeleton, or unrelated."""
    ca, cb = canonicalize(smiles_a), canonicalize(smiles_b)
    out = {
        "same_molecule": bool(ca and ca == cb),
        "same_skeleton": False,
        "same_formula": False,
        "canonical_a": ca,
        "canonical_b": cb,
    }
    if ca is None or cb is None:
        return out
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    out["same_skeleton"] = (
        Chem.MolToInchiKey(ma).split("-")[0] == Chem.MolToInchiKey(mb).split("-")[0]
    )
    out["same_formula"] = rdMolDescriptors.CalcMolFormula(
        ma
    ) == rdMolDescriptors.CalcMolFormula(mb)
    return out


# ---------------------------------------------------------------------------
# Consensus and calibrated confidence
# ---------------------------------------------------------------------------


def vote(results: list[dict], priority: list[str] | None = None) -> dict:
    """Group specialist predictions and report the agreement pattern.

    Predictions are canonicalized before grouping, so the same molecule written two
    ways counts once. See :func:`canonicalize` for why that is not optional.

    Distinguishes two things the caller must not conflate. ``committee`` is every
    model that was asked, and must match the calibration table. ``voters`` is the
    subset that returned a parseable SMILES.

    An abstention counts as a dissenting vote, not as absence. A model that ran and
    produced garbage contributes one singleton to the pattern, so the pattern's parts
    always sum to the committee size: four models can only ever produce "4", "3/1",
    "2/1/1", "2/2", or "1/1/1/1".

    The alternative -- dropping abstainers and shrinking the pattern -- was measured
    on 722 benchmark items and rejected. It split the same situations across eleven
    buckets instead of five, pushing 12% of items below the sample floor where no
    number can be quoted (2% under this rule). It also mislabelled the evidence: a
    three-way agreement with one model unable to read the image at all was recorded
    as "3", which reads like consensus, when the abstention is itself a signal that
    the image is hard. Under this rule that item is "3/1", and the empirical accuracy
    of the merged bucket bears this out.

    Parameters
    ----------
    results : list[dict]
        One entry per model: ``{"model", "smiles", "ok", "error", "infer_s"}``.
        ``model`` is a bare name with no "local:" prefix.
    priority : list[str], optional
        Tie-break order, strongest model first. Defaults to the order given.

    Returns
    -------
    dict
        pattern, winner, votes, abstained, committee, voters
    """
    # Normalize the "local:" prefix here rather than trusting every caller. OCSR's
    # LocalOCSRClient dispatches on ModelSpec names like "local:decimer", while the
    # calibration table's committee is bare names. A mismatch does not raise: it makes
    # check_committee report committee_mismatch, which nulls the confidence for every
    # ensemble call. One missing removeprefix would turn the feature off permanently
    # behind a plausible-looking error, so it is not left to the caller.
    def _bare(name: str) -> str:
        return name.removeprefix("local:")

    committee = [_bare(r.get("model") or "") for r in results]
    order = [_bare(m) for m in (priority or committee)]

    groups: dict[str, list[str]] = {}
    abstained: dict[str, str] = {}
    # Counted separately from the dict: two results carrying the same model name, which
    # `--models decimer,decimer` and the "local:" alias both produce, collapse to one
    # dict key and would silently drop a singleton from the pattern.
    n_abstained = 0
    for r in results:
        model = _bare(r.get("model") or "")
        canon = canonicalize(r.get("smiles")) if r.get("ok", True) else None
        if canon is None:
            abstained[model] = str(r.get("smiles") or r.get("error") or "")[:80]
            n_abstained += 1
            continue
        groups.setdefault(canon, []).append(model)

    if not groups:
        # Every model failed. There is no winner to be right about, but the item is
        # not absent from the evidence either: it is the worst case of disagreement.
        # Reporting it as all-singletons keeps it in the table, where it counts
        # against that bucket, instead of vanishing and flattering every other row.
        return {
            "pattern": "/".join("1" * len(committee)) if committee else None,
            "winner": None,
            "votes": {},
            "abstained": abstained,
            "committee": committee,
            "voters": [],
        }

    # Each abstainer is its own singleton: it agrees with nobody, including other
    # abstainers, whose garbage output is unrelated.
    counts = sorted(
        [len(v) for v in groups.values()] + [1] * n_abstained, reverse=True
    )
    top = max(len(v) for v in groups.values())
    tied = [smi for smi, models in groups.items() if len(models) == top]
    if len(tied) > 1:
        # Deterministic tie-break by model priority. This choice is load-bearing:
        # it makes an all-different pattern report the strongest model's answer, so
        # that bucket's accuracy in a calibration table is really that model's solo
        # accuracy. Any table must record which rule was used.
        #
        # Resolved over the groups, so only models that actually voted can win. An
        # earlier version scanned `results` and never re-checked ok, which let a model
        # whose output failed to parse still decide the answer whenever its unusable
        # string happened to canonicalize into a tied group. The winner would then be
        # a SMILES from outside `voters`, and the bucket would not carry the solo
        # accuracy the docstring above claims for it.
        winner = next(
            (smi for m in order for smi in tied if m in groups[smi]),
            tied[0],
        )
    else:
        winner = tied[0]

    return {
        "pattern": "/".join(str(c) for c in counts),
        "winner": winner,
        "votes": groups,
        "abstained": abstained,
        "committee": committee,
        "voters": [m for models in groups.values() for m in models],
    }


def _label_for(p: float, low_n: bool = False) -> str:
    """Map a probability to a coarse label, prefixed when the bucket is small."""
    name = "conflicting"
    for threshold, band in _LABEL_BANDS:
        if p >= threshold:
            name = band
            break
    return f"low_n_{name}" if low_n else name


def confidence(pattern: str | None, table: dict) -> dict:
    """Look up P(majority correct) for an agreement pattern.

    Takes the table as an argument rather than reading a module global, so the tool
    and the recalibration script share one code path.

    Reports a point estimate only where the bucket cleared the table's sample floor.
    Below it the 95% interval spans 30-60 pp, so a decimal would be false precision;
    the label and the interval still carry the actionable part. An unknown pattern
    gets no number at all rather than a guess.

    Returns
    -------
    dict
        p, label, n, ci, reason
    """
    if pattern is None:
        return {"p": None, "label": "unavailable", "n": 0, "ci": None,
                "reason": "no_prediction"}

    entry = (table.get("patterns") or {}).get(pattern)
    if entry is None:
        return {"p": None, "label": "unknown", "n": 0, "ci": None,
                "reason": "unknown_pattern"}

    p, n, ci = entry.get("p"), entry.get("n", 0), entry.get("ci")
    if p is None:
        # Below the floor we withhold the number but still owe a useful label, so it
        # is derived from the Jeffreys estimate: the same quantity the `p` column
        # reports above the floor, which keeps one rule across the whole table.
        #
        # The two obvious alternatives both mislead. The raw point estimate k/n calls
        # a 7/7 bucket "unanimous", claiming certainty from seven items. The
        # interval's lower bound calls that same bucket "conflicting", overstating
        # the doubt: seven for seven is thin evidence, and it points one way.
        # Jeffreys shrinks toward 0.5 in proportion to how thin the bucket is,
        # putting 7/7 at 0.938 and so at low_n_weak, one band below the 0.95 cut.
        k = entry.get("k")
        est = ((k + 0.5) / (n + 1.0)) if (k is not None and n) else (ci[0] if ci else 0.0)
        label = entry.get("label") or _label_for(est, low_n=True)
        return {"p": None, "label": label, "n": n, "ci": ci,
                "reason": "below_n_floor"}
    return {"p": p, "label": entry.get("label") or _label_for(p), "n": n, "ci": ci,
            "reason": None}


def prior_confidence(model: str, table: dict | None = None) -> dict:
    """Confidence for a single-model backend, from that model's measured solo accuracy.

    A single model has no consensus to measure, so the agreement table does not apply.
    Do **not** route a one-model result through :func:`vote` and :func:`confidence`:
    that yields pattern "1", which a four-model table does not contain at all, so the
    lookup misses and reports unknown_pattern. It would also trip
    :func:`check_committee`. Use this instead.

    Returns the same shape as :func:`confidence` so callers have one code path.

    Every number here comes from the calibration table's ``model_performance`` section, never from
    a constant in this file. Measurements belong with the data that produced them: a
    user who refits on their own images gets priors for their images, and a stale
    figure cannot outlive the table it was measured on. The registry in ocsr_models
    describes what a model *is* (how to run it, how fast, what it needs); the table
    records how well it *did*.

    Passing ``table`` lets a caller that already loaded one avoid a second read.
    """
    bare = model.removeprefix("local:")
    try:
        t = table if table is not None else load_calibration()
    except (OSError, ValueError, TypeError) as exc:
        return {"p": None, "label": "unavailable", "n": 0, "ci": None,
                "reason": f"calibration_unreadable: {type(exc).__name__}"}

    entry = (t.get("model_performance") or {}).get(bare)
    if not entry or entry.get("accuracy") is None:
        return {"p": None, "label": "unavailable", "n": 0, "ci": None,
                "reason": "no_prior_for_model"}
    p = entry["accuracy"]
    return {"p": p, "label": _label_for(p), "n": entry.get("n", 0),
            "ci": entry.get("ci"), "reason": None}


def model_performance(model: str, table: dict | None = None) -> dict:
    """Measured single-model performance from a calibration table, or {} if absent.

    The one place to ask "how good is this model": accuracy, the k and n behind it,
    a 95% interval, and how often it abstains. Nothing here is compiled into the
    source, so a table refitted on other images reports that table's numbers.
    """
    try:
        t = table if table is not None else load_calibration()
    except (OSError, ValueError, TypeError):
        return {}
    return dict((t.get("model_performance") or {}).get(model.removeprefix("local:")) or {})


def load_calibration(path: str | None = None) -> dict:
    """Load a calibration table: explicit path, then env var, then the packaged default.

    The packaged default describes the four-model committee on RDKit-rendered images
    and is what makes ``backend="ensemble"`` work with no configuration. Anyone who
    builds their own table with the recalibration script points at it here.

    A table drives which answers get a confidence, so a malformed one is rejected on
    load rather than silently producing wrong numbers deep in a run.
    """
    candidate = path or os.environ.get("CHEMGRAPH_OCSR_CALIBRATION")
    if candidate:
        resolved = os.path.expanduser(candidate)
        # A FIFO here blocks open() forever with no caller-side timeout, the same
        # reason load_image_bytes checks. Callers of this function have usually just
        # spent real inference time and must not hang holding the result.
        if not stat.S_ISREG(os.lstat(resolved).st_mode):
            raise ValueError(f"calibration table is not a regular file: {resolved}")
        with open(resolved) as fh:
            return _validate_calibration(_load_json(fh, resolved), resolved)

    from importlib import resources

    # importlib.resources, not a __file__-relative path: the latter works under
    # `pip install -e` and silently fails on a normal install.
    ref = resources.files("chemgraph.tools").joinpath("ocsr_calibration_4model.json")
    with ref.open() as fh:
        return _validate_calibration(_load_json(fh, "packaged default"), "packaged default")


def _load_json(fh, origin: str) -> object:
    """json.load, with every failure normalised to ValueError.

    Callers guard on (OSError, ValueError, TypeError) so that a bad table costs the
    confidence and not the prediction. json.load can also raise RecursionError on a
    deeply nested file, which slipped past all of them and discarded a completed
    ensemble run.
    """
    try:
        return json.load(fh)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not parse {origin}: {type(exc).__name__}") from None


def _validate_calibration(table: object, origin: str) -> dict:
    """Reject a table that cannot mean what a caller will assume it means."""
    def bad(why: str) -> ValueError:
        return ValueError(f"calibration table {origin!r} is unusable: {why}")

    # Arithmetic below converts to float, and an int with hundreds of digits raises
    # OverflowError, which is neither ValueError nor TypeError and so escaped every
    # guard in the call chain.
    def num(value: object, field: str) -> float:
        try:
            return float(value)
        except (OverflowError, ValueError, TypeError):
            raise bad(f"{field} is not a usable number: {value!r}") from None

    if not isinstance(table, dict):
        raise bad(f"top level is {type(table).__name__}, expected an object")
    committee = table.get("committee")
    if not isinstance(committee, list) or not all(isinstance(m, str) for m in committee):
        raise bad("'committee' must be a list of model names")
    # An empty committee disables check_committee, which only compares when both sides
    # are non-empty, so every mismatch would pass and any table would apply anywhere.
    if not committee:
        raise bad("'committee' is empty, which would disable the committee check")
    # check_committee compares sorted name lists, so a duplicate makes a table that
    # can never match any real run and reports committee_mismatch forever.
    if len(set(committee)) != len(committee):
        raise bad(f"'committee' repeats a model: {committee}")
    # A tie_break that does not parse, or that does not name exactly the committee,
    # would fall back to the committee's arbitrary JSON order: the tool would vote one
    # way and quote a number measured the other way, with nothing in the result to
    # show it. Reject at load instead.
    recorded = table.get("tie_break")
    if recorded is not None:
        order = _parse_tie_break(recorded)
        if sorted(order) != sorted(committee):
            raise bad(
                f"'tie_break' must name exactly the committee, most accurate first. "
                f"Committee: {committee}. Parsed from tie_break: {order}. Expected "
                f"the form 'model-priority: a,b,c'."
            )

    patterns = table.get("patterns")
    if not isinstance(patterns, dict) or not patterns:
        raise bad("'patterns' must be a non-empty object")

    size = len(committee)
    for name, cell in patterns.items():
        if not isinstance(cell, dict):
            raise bad(f"pattern {name!r} is not an object")
        try:
            parts = [int(x) for x in str(name).split("/")]
        except ValueError:
            raise bad(f"pattern {name!r} is not slash-separated integers") from None
        # int() accepts a leading minus and unicode digits, so the sum check below is
        # not enough on its own: "2/-1" sums to 1 and would pass for a one-model
        # committee. A part is a count of models agreeing, so it is at least 1.
        if any(p < 1 for p in parts):
            raise bad(f"pattern {name!r} has a part below 1; each part counts models")
        # A pattern whose parts do not sum to the committee size means the table was
        # fit under a different abstention rule; its buckets would not line up with
        # what vote() produces, and every lookup would quietly miss or mismatch.
        if sum(parts) != size:
            raise bad(f"pattern {name!r} sums to {sum(parts)}, committee has {size} models")
        k, n = cell.get("k"), cell.get("n")
        if (isinstance(k, bool) or isinstance(n, bool)
                or not isinstance(n, int) or n < 0
                or not isinstance(k, int) or not 0 <= k <= n):
            raise bad(f"pattern {name!r} has invalid k/n: {k!r}/{n!r}")
        num(k, f"pattern {name!r} k")
        num(n, f"pattern {name!r} n")
        # A quotable number from no observations at all: (0+0.5)/(0+1) == 0.5 makes
        # the consistency check below pass, so it has to be caught here.
        if n == 0 and cell.get("p") is not None:
            raise bad(f"pattern {name!r} quotes p over 0 observations")

        # p, ci and label are what a caller acts on, so check them here. Left
        # unchecked, a string p reached _label_for and raised TypeError deep inside a
        # lookup, and a p that simply disagreed with its own k and n was returned as
        # a confident answer with nothing to reveal the contradiction.
        p = cell.get("p")
        if p is not None:
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                raise bad(f"pattern {name!r} has a non-numeric p: {p!r}")
            if not 0.0 <= p <= 1.0:
                raise bad(f"pattern {name!r} has p={p} outside [0, 1]")
            expected = round((k + 0.5) / (n + 1.0), 4)
            if abs(p - expected) > 5e-4:
                raise bad(
                    f"pattern {name!r} says p={p} but k/n = {k}/{n} gives {expected}. "
                    f"Refit with chemgraph.tools.ocsr_calibrate instead of editing p."
                )
        ci = cell.get("ci")
        if ci is not None:
            ok = (isinstance(ci, list) and len(ci) == 2
                  and all(isinstance(b, (int, float)) and not isinstance(b, bool)
                          and math.isfinite(num(b, f"pattern {name!r} ci"))
                          and 0.0 <= b <= 1.0
                          for b in ci))
            if not ok:
                # json.loads accepts NaN and Infinity, and a NaN bound compares false
                # against everything, so a low-n label would silently come out wrong.
                raise bad(f"pattern {name!r} has a malformed ci: {ci!r}")
            if ci[0] > ci[1]:
                raise bad(f"pattern {name!r} has a reversed ci: {ci!r}")
        label = cell.get("label")
        if label is not None and not isinstance(label, str):
            raise bad(f"pattern {name!r} has a non-string label: {label!r}")

    # Checked here so prior_confidence and model_performance can read it without
    # each guarding separately. A string accuracy used to reach _label_for and raise
    # TypeError from inside a lookup, and mock_ocsr inherited the crash.
    performance = table.get("model_performance")
    if performance is not None:
        if not isinstance(performance, dict):
            raise bad("'model_performance' must be an object keyed by model name")
        for name, entry in performance.items():
            if not isinstance(entry, dict):
                raise bad(f"model_performance.{name} is not an object")
            accuracy = entry.get("accuracy")
            if accuracy is None:
                continue  # unmeasured is allowed; it reports no_prior_for_model
            if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
                raise bad(f"model_performance.{name} has a non-numeric accuracy")
            if not 0.0 <= accuracy <= 1.0:
                raise bad(f"model_performance.{name} has accuracy={accuracy} "
                          f"outside [0, 1]")
            count = entry.get("n")
            if count is not None and (isinstance(count, bool)
                                      or not isinstance(count, int) or count < 0):
                raise bad(f"model_performance.{name} has an invalid n: {count!r}")
            # An accuracy backed by nothing is worse than no accuracy: it reads as a
            # real measurement and there is no field left to signal otherwise.
            if count == 0:
                raise bad(f"model_performance.{name} reports an accuracy over 0 "
                          f"observations; drop the entry or set accuracy to null")
    return table


def tie_break_order(table: dict) -> list[str]:
    """The model priority a table was fitted under, from its own tie_break field.

    The order decides which answer wins a tie, so it decides what the all-different
    bucket's accuracy actually measures. Using the registry's order instead of the
    table's would attach a number measured for one model's answer to a different
    model's answer, and check_committee cannot see it because it compares sorted
    names.

    A table with no tie_break falls back to its committee order, which is the best
    available guess for a table written before the field existed. A table whose
    tie_break is present but unusable never reaches here: _validate_calibration
    rejects it, because falling back would silently vote one way while quoting a
    number measured the other way.
    """
    order = _parse_tie_break(table.get("tie_break"))
    return order if order is not None else list(table.get("committee") or [])


def _parse_tie_break(value: object) -> list[str] | None:
    """Model names from a tie_break string, or None when there is nothing to parse.

    Returns None for an absent field and for one that does not parse, so a caller can
    tell "no preference recorded" from "recorded something I cannot read".
    """
    if value is None:
        return None
    if not isinstance(value, str) or "model-priority:" not in value:
        return []  # present and unusable: an empty list, never a silent fallback
    _, _, names = value.partition("model-priority:")
    return [n.strip() for n in names.split(",") if n.strip()]


def check_committee(vote_result: dict, table: dict) -> str | None:
    """Return a reason string if the table does not describe this committee, else None.

    Compares the models that were *asked*, not the ones that voted, since abstention
    is normal and does not change which table applies.
    """
    ran = sorted(vote_result.get("committee") or [])
    fit = sorted(table.get("committee") or [])
    if not ran or not fit or ran == fit:
        return None

    # Name the missing models and how to get them. A partial install is the common
    # case here, and "committee_mismatch: [...] vs [...]" tells a user their answer
    # has no confidence without telling them what to do about it.
    absent = [m for m in fit if m not in ran]
    extra = [m for m in ran if m not in fit]
    if absent and not extra:
        # shlex.quote because these names come from a JSON file and the result is a
        # command a user or an agent will copy and paste.
        import shlex

        install = "; ".join(
            f"python -m chemgraph.tools.ocsr_setup {shlex.quote(m)}" for m in absent
        )
        return (
            f"committee_mismatch: the table was fit on {fit} and this machine ran "
            f"{ran}, so no calibrated number applies. Install the rest with: "
            f"{install}; or refit for the models you have with "
            f"python -m chemgraph.tools.ocsr_calibrate --labels YOUR_LABELS.csv "
            f"--models {shlex.quote(','.join(ran))}"
        )
    if extra and not absent:
        return (
            f"committee_mismatch: the table was fit on {fit} and this machine ran "
            f"the larger set {ran}. Vote only the committee the table describes with "
            f"models_wanted={fit}."
        )
    # Both sets differ. Neither installing nor subsetting alone fixes it, so name the
    # only two things that do.
    return (
        f"committee_mismatch: the table was fit on {fit} and this machine ran {ran}, "
        f"which is a different set, so no calibrated number applies. Either install "
        f"{', '.join(absent)} and vote the table's committee, or refit for what you "
        f"have with python -m chemgraph.tools.ocsr_calibrate --labels YOUR_LABELS.csv "
        f"--models {','.join(ran)}"
    )


# ---------------------------------------------------------------------------
# Hermetic test helper
# ---------------------------------------------------------------------------


def build_result(**overrides) -> dict:
    """Assemble the tool's return dict, with every contract key present.

    One place builds this shape, so a backend cannot ship a partial dict and an agent
    never has to test for a missing key. The docstring of ``image_to_smiles`` promises
    these exact fields; changing the set means changing both together.

    ``votes`` and ``abstained`` stay None outside ``backend="ensemble"``, where a
    single model has nothing to report.
    """
    result = {
        "ok": False,
        "smiles": None,
        "valid": False,
        "formula": None,
        "n_fragments": 0,
        "confidence": None,
        "confidence_label": "unavailable",
        "confidence_unavailable_reason": None,
        "agreement": None,
        "basis": None,
        "backend_used": None,
        "model_used": None,
        "cold_start": False,
        "latency_s": 0.0,
        "error": "",
        "warning": "",
        "votes": None,
        "abstained": None,
    }
    unknown = set(overrides) - set(result)
    if unknown:
        raise KeyError(f"not part of the OCSR result contract: {sorted(unknown)}")
    result.update(overrides)
    return result


def mock_ocsr(image_path: str, backend: str = "mock") -> dict:
    """Deterministic stand-in for a real backend, for tests with no models installed.

    Mirrors :func:`chemgraph.tools.docking_core.mock_docking`. Returns aspirin for
    any input, so a test can assert on the plumbing without waiting for a model load.
    Goes through :func:`build_result`, so a test written against it is testing the
    real contract.

    It reports no confidence, matching what a single model returns: one model cannot
    say how likely it is to be right about a particular image. Returning a number
    here would make the helper model a shape no backend produces, which is the one
    thing a stand-in must not do.
    """
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    v = validate_smiles_core(smiles)
    return build_result(
        ok=True,
        smiles=smiles,
        valid=v["valid"],
        formula=v["formula"],
        n_fragments=v["n_fragments"],
        confidence=None,
        confidence_label="unavailable",
        confidence_unavailable_reason="single_model_has_no_per_image_confidence",
        agreement="single",
        backend_used=backend,
        model_used="mock",
    )
