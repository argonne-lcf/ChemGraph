"""Pure-Python OCSR helpers (no LangChain / MCP decorators).

Optical Chemical Structure Recognition: read a molecule's 2D structure diagram from
an image and return its SMILES. Used by the LangChain ``@tool`` wrappers in
:mod:`chemgraph.tools.ocsr_tools`.

Everything here is pure: image sniffing and loading, pulling a SMILES out of model
output, and validating it with RDKit. Nothing in this module runs a model, opens a
socket or touches a checkpoint, so it imports and its tests collect on a host with
no models installed. :mod:`chemgraph.tools.ocsr_backends` holds the parts that do.

RDKit is a core ChemGraph dependency and is imported lazily here regardless.
"""

from __future__ import annotations

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

# RDKit parsing is superlinear in string length: 20k characters costs about 9 s of
# CPU and a megabyte-scale string is effectively unbounded. Model output is untrusted,
# so cap it before RDKit sees it. Real SMILES are well under 1000 characters; the
# longest reference label in the OCSR benchmark is 152 and the longest prediction
# any model returned is 485.
_MAX_SMILES_CHARS = 4000

# Cap for the free-text fields of a result. Each can echo a caller-supplied path,
# model name or backend message, and the dict is read back into an agent's context.
# 1200 clears the longest legitimate warning, which is four caveats at once: a
# salt, the models that could not run, a committee the table does not describe,
# and a stereo split. Searching that space by driving image_to_smiles_core puts
# the maximum near 1190. No per-term breakdown here: it was written four times
# and wrong four times, because the terms trade off against each other through the
# number of models running and cannot each be taken at their own maximum. A lower
# cap drops the refit command off the end of the mismatch note, which is the one
# part of that message a user has to act on.
_MAX_MESSAGE_CHARS = 1200

# Confidence label cut-points, applied to the point estimate. Below the calibration
# table's n floor these are prefixed "low_n_", because a 30-60 pp interval does not
# deserve a decimal.
_LABEL_BANDS = ((0.99, "unanimous"), (0.95, "strong"), (0.70, "weak"))


# The optional quotes around the key let a JSON reply be read at the line level:
# '"smiles": "CCO"' is the single most common structured form a vision LLM returns,
# and reaching it here keeps the word-by-word fallback from matching an element
# symbol out of an adjacent "elements" list first.
# Matches a "smiles": "..." pair anywhere, so a one-line JSON reply is read from its
# own field instead of from whatever token happens to parse first.
_JSON_SMILES_RE = re.compile(r"[\"'](?:smiles)[\"']\s*:\s*[\"']([^\"']*)[\"']",
                             re.IGNORECASE)

_SMILES_LABEL_RE = re.compile(
    r"^\s*[\"']?(?:smiles|answer|result)[\"']?\s*[:=]\s*", re.IGNORECASE)


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


def resolve_image_path(image_path: str) -> str:
    """Expand ``~`` and resolve a bare name the way every other file-reading tool does.

    A relative path is looked up in ``CHEMGRAPH_LOG_DIR`` when it is not found in the
    working directory, so an image an agent wrote there can be read back by bare name.
    Matches ``ase_core._resolve_existing_path``, and repeats its three lines instead of
    importing it: that module reaches ``torch`` through the fairchem calculator schema,
    which would put a torch import back on the path of every OCSR call.
    """
    path = os.path.expanduser(image_path)
    if not os.path.isfile(path):
        log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
        if log_dir and not os.path.isabs(path):
            candidate = os.path.join(log_dir, path)
            if os.path.isfile(candidate):
                path = candidate
    return os.path.abspath(path)


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
    path = resolve_image_path(image_path)
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
        raise ValueError(f"image is {st.st_size} bytes, over the {max_bytes} limit: "
                         f"{echo(path)}")

    with open(path, "rb") as fh:
        data = fh.read(max_bytes + 1)  # +1 so a file that grew since lstat is caught
    if len(data) > max_bytes:
        raise ValueError(f"image grew past the {max_bytes} limit while reading: "
                         f"{echo(path)}")

    mime = _sniff_mime(data[:16])
    if mime is None:
        raise ValueError(
            f"not a recognized image (PNG/JPEG/GIF/WEBP) by content: {path}. "
            f"The extension is not trusted; only the leading bytes are."
        )
    return data, mime


# ---------------------------------------------------------------------------
# SMILES handling
# ---------------------------------------------------------------------------


def _encodable(smiles: str) -> bool:
    """True when the string can cross into RDKit, which takes bytes.

    json.loads accepts a lone surrogate such as ``\\ud800``, so a model reply
    carrying a broken UTF-16 pair arrives here as a str no encoder can render.
    Every RDKit entry point then raises UnicodeEncodeError, which is a raise out of
    two functions whose contract is to return a value for any input.
    """
    try:
        smiles.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def jeffreys(k: int, n: int) -> float:
    """Posterior mean under Beta(0.5, 0.5). Keeps a perfect bucket below 1.0.

    One definition, because the table records this as the rule covering every
    number in it and the validator enforces that. ocsr_calibrate imports it, so a
    table cannot be fitted under one formula and checked against another.
    """
    return (k + 0.5) / (n + 1.0)


def canonicalize(smiles: str | None, stereo: bool = False) -> str | None:
    """Canonical SMILES, or None when the string does not parse.

    Defaults to stereo-blind, matching how the OCSR benchmark scores: most reference
    labels carry no stereochemistry, so comparing with it makes a model that correctly
    reads a wedge bond look wrong.

    ``stereo=True`` is what the tool returns to the caller, so two models that read
    the same structure produce the same string: DECIMER emits Kekule SMILES while the
    other three emit aromatic, and on 422 benchmark items raw-string comparison finds
    the four unanimous on 12 where canonical comparison finds it on 289.
    """
    if not smiles or len(smiles) > _MAX_SMILES_CHARS or not _encodable(smiles):
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


_WRAPPER_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"))


def _strip_wrappers(s: str) -> str:
    """Peel markdown emphasis, quotes and balanced brackets off a candidate.

    Emphasis is the dangerous case. ``*`` is RDKit's wildcard atom, so ``**CCO**``
    parses as a valid five-atom molecule and would be returned as the prediction with
    two extra atoms in it. Quotes and brackets only cost a match, since ``(CCO)``
    fails to parse.

    Brackets are peeled only when balanced across the whole string, because a SMILES
    legitimately contains them: ``[Na+]`` and ``C(=O)O`` must survive unchanged.
    """
    s = s.strip()
    while True:
        before = s
        s = s.strip("`\"'").strip()
        while len(s) > 1 and s[0] == s[-1] == "*":
            s = s[1:-1].strip()
        for opener, closer in _WRAPPER_PAIRS:
            if len(s) > 1 and s[0] == opener and s[-1] == closer:
                inner = s[1:-1]
                # Only unwrap an outer pair that encloses the whole string, so
                # "C(=O)O" is left alone: its first "(" closes before the end.
                depth = 0
                encloses = True
                for i, ch in enumerate(s):
                    depth += (ch == opener) - (ch == closer)
                    if depth == 0 and i < len(s) - 1:
                        encloses = False
                        break
                # Keep the wrapped form when it is the one that parses: "[NH4+]" is
                # a whole SMILES, and stripping its brackets leaves "NH4+", which is
                # not. Unwrap only when doing so turns a non-molecule into one.
                if encloses and canonicalize(s) is None:
                    s = inner.strip()
        if s == before:
            return s


def extract_smiles(raw_text: str | None) -> str | None:
    """Pull a SMILES out of a model's raw reply.

    Handles a bare string, a fenced code block, and "The SMILES is: X" prose. Returns
    None for a refusal ("I cannot process images"), which must not be mistaken for a
    prediction. Does not validate; :func:`validate_smiles_core` does that.
    """
    if not raw_text or len(raw_text) > _MAX_SMILES_CHARS:
        return None
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw_text).replace("```", "")
    # A JSON object on one line never reaches the line-anchored label below, and the
    # word-by-word fallback would take "C" out of an adjacent "elements" list first.
    keyed = _JSON_SMILES_RE.search(text)
    if keyed:
        candidate = keyed.group(1).strip()
        if candidate and not _looks_like_prose(candidate) and canonicalize(candidate):
            return candidate
    for line in text.splitlines():
        cleaned = _strip_wrappers(_SMILES_LABEL_RE.sub("", line))
        if cleaned and not _looks_like_prose(cleaned) and canonicalize(cleaned) is not None:
            return cleaned
    for token in text.split():
        cleaned = _strip_wrappers(token.rstrip(".,;!?"))
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
    if not _encodable(smiles):
        out["errors"].append("SMILES contains characters that are not valid text")
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


# ---------------------------------------------------------------------------
# The tool's return contract
# ---------------------------------------------------------------------------


def echo(value: object, limit: int = 120) -> str:
    """A caller-supplied value, shortened for quoting in a message.

    Bounds the echo where it happens. The cap in :func:`build_result` bounds the
    assembled field, which has to stay wide enough for four legitimate caveats at
    once, so on its own it lets one 50,000-character path fill the whole budget and
    push the part a user has to act on off the end.

    Cut from the middle. What is quoted here is usually a path, and a deep directory
    chain is identical across every file in it, so trimming the tail leaves two
    different images with byte-identical errors and an agent cannot tell which one
    failed.
    """
    text = str(value)
    if len(text) <= limit:
        return text
    head = (limit - 3) // 2
    return text[:head] + "..." + text[len(text) - (limit - 3 - head):]


def build_result(**overrides) -> dict:
    """Assemble the tool's return dict, with every contract key present.

    One place builds this shape, so a backend cannot ship a partial dict and an agent
    never has to test for a missing key. The docstring of ``image_to_smiles`` promises
    these exact fields; changing the set means changing both together.

    The committee keys stay None outside an ensemble call, where a single model has
    nothing to report: it produced one answer and there is no agreement to measure.
    A single model still carries no per-image confidence, so ``confidence`` is None
    there and ``confidence_unavailable_reason`` says why; :func:`prior_confidence`
    is the way to ask how accurate that model is in general.
    """
    result = {
        "ok": False,
        "smiles": None,
        "valid": False,
        "formula": None,
        "n_fragments": 0,
        "confidence": None,
        "confidence_interval": None,
        "confidence_label": "unavailable",
        "confidence_unavailable_reason": None,
        "agreement": None,
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
    # Cap the free text here rather than at each site that builds it. Every one of
    # them can echo something the caller supplied, a path, a model name, a backend
    # message, and the whole dict goes back into an agent's context. One place to
    # enforce it means a new message cannot reintroduce an unbounded field.
    for key in ("error", "warning", "confidence_unavailable_reason"):
        text = result[key]
        if isinstance(text, str) and len(text) > _MAX_MESSAGE_CHARS:
            result[key] = text[:_MAX_MESSAGE_CHARS - 3] + "..."
    return result


def fragment_warning(validation: dict) -> str:
    """The caveat for a result holding more than one molecule, or "" for one.

    Both backends return it, so the text lives here: an agent that learns to act on
    the single-model wording must see the same words from a committee.
    """
    n = validation.get("n_fragments", 0)
    if n <= 1:
        return ""
    return (f"the image contains {n} disconnected fragments (a salt, a mixture, or "
            f"a reaction scheme). Ask which one is meant before using this SMILES.")


def vote(results: list[dict], priority: list[str] | None = None) -> dict:
    """Group specialist predictions and report the agreement pattern.

    Predictions are canonicalized before grouping, so the same molecule written two
    ways counts once.

    ``committee`` is every model that was asked; ``voters`` is the subset that
    returned a parseable SMILES. An abstention counts as a dissenting vote: a model
    that ran and produced garbage contributes one singleton, so the pattern's parts
    always sum to the committee size. Four models can only ever produce "4", "3/1",
    "2/1/1", "2/2", or "1/1/1/1".

    Dropping abstainers and shrinking the pattern was measured on 722 benchmark items
    and rejected. It split the same situations across eleven buckets instead of five,
    putting 12% of items below the sample floor where no number can be quoted, against
    2% under this rule.

    Parameters
    ----------
    results : list[dict]
        One entry per model: ``{"model", "smiles", "ok"}``.
    priority : list[str], optional
        Tie-break order, strongest model first. Defaults to the order given.

    Returns
    -------
    dict
        pattern, winner, votes, abstained, committee, voters
    """
    def _bare(name: str) -> str:
        return name.removeprefix("local:")

    committee = [_bare(r.get("model") or "") for r in results]
    order = [_bare(m) for m in (priority or committee)]

    groups: dict[str, list[str]] = {}
    abstained: dict[str, str] = {}
    # Counted separately from the dict: two results carrying the same model name
    # collapse to one dict key and would drop a singleton from the pattern.
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
        # Every model failed. Reporting all-singletons keeps the item in the evidence,
        # where it counts against that bucket, instead of flattering every other row.
        return {
            "pattern": "/".join("1" * len(committee)) if committee else None,
            "winner": None,
            "votes": {},
            "abstained": abstained,
            "committee": committee,
            "voters": [],
        }

    counts = sorted([len(v) for v in groups.values()] + [1] * n_abstained, reverse=True)
    top = max(len(v) for v in groups.values())
    tied = [smi for smi, models in groups.items() if len(models) == top]
    if len(tied) > 1:
        # Deterministic tie-break by model priority, so an all-different pattern
        # reports the strongest model's answer and that bucket's measured accuracy is
        # really that model's solo accuracy. Resolved over the groups, so a model
        # whose output failed to parse cannot decide the answer.
        winner = next((smi for m in order for smi in tied if m in groups[smi]), tied[0])
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
        # Before expanduser, which calls __fspath__ on anything that has one. Every
        # caller guards this function for OSError, ValueError and TypeError, and an
        # object whose __fspath__ raises something else escapes all three. A Path is
        # the ordinary way to pass a path, so it is converted here where whatever
        # __fspath__ raises can be turned into one of the three.
        if not isinstance(candidate, (str, os.PathLike)):
            raise ValueError(f"calibration must be a path, got "
                             f"{type(candidate).__name__}")
        try:
            candidate = os.fspath(candidate)
        except Exception as exc:
            raise ValueError(f"calibration path is unusable: "
                             f"{type(exc).__name__}") from None
        resolved = os.path.expanduser(candidate)
        # A FIFO here blocks open() forever with no caller-side timeout, the same
        # reason load_image_bytes checks. Callers of this function have usually just
        # spent real inference time and must not hang holding the result.
        if not stat.S_ISREG(os.lstat(resolved).st_mode):
            raise ValueError(f"calibration table is not a regular file: {echo(resolved)}")
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

    def check_ci(span: object, where: str) -> None:
        """A pattern cell and a model_performance entry bound theirs identically.

        isfinite because json.loads accepts NaN and Infinity, and a NaN bound
        compares false against everything, so a low-n label would come out wrong
        with nothing to show why.
        """
        if span is None:
            return
        ok = (isinstance(span, list) and len(span) == 2
              and all(isinstance(b, (int, float)) and not isinstance(b, bool)
                      and math.isfinite(num(b, f"{where} ci")) and 0.0 <= b <= 1.0
                      for b in span))
        if not ok:
            raise bad(f"{where} has a malformed ci: {span!r}")
        if span[0] > span[1]:
            raise bad(f"{where} has a reversed ci: {span!r}")

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

    # Checked because confidence() acts on it: an unusable value silently disables
    # the floor, and the table would quote a point estimate from a handful of images
    # while recording that it does not.
    declared = table.get("min_n_for_point_estimate")
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, (int, float)):
            raise bad(f"'min_n_for_point_estimate' must be a non-negative number: "
                      f"{declared!r}")
        # Through num(), so a 400-digit int raises ValueError here rather than
        # OverflowError from the isfinite call, which no caller guards against.
        # isfinite for the reason the ci check uses it: json accepts NaN, which
        # compares false against everything so the floor would never fire, and
        # Infinity, which fires on every bucket however large.
        if not math.isfinite(num(declared, "'min_n_for_point_estimate'")) or declared < 0:
            raise bad(f"'min_n_for_point_estimate' must be a non-negative number: "
                      f"{declared!r}")

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
        # vote() emits counts largest first, joined by a slash and nothing else, so
        # a key that differs from that in any way is a bucket no lookup can ever
        # reach. Descending order is one way to differ. int() also normalises, so
        # "+4", "0004", " 4 " and the Arabic-Indic digit all parse to the same parts
        # and none of them is ever looked up. Rejecting the key beats shipping a
        # silently dead row.
        canonical = "/".join(str(x) for x in sorted(parts, reverse=True))
        if str(name) != canonical:
            raise bad(f"pattern {name!r} is not the key vote() emits, which is "
                      f"{canonical!r}")
        k, n = cell.get("k"), cell.get("n")
        if (isinstance(k, bool) or isinstance(n, bool)
                or not isinstance(n, int) or n < 0
                or not isinstance(k, int) or not 0 <= k <= n):
            raise bad(f"pattern {name!r} has invalid k/n: {k!r}/{n!r}")
        # Not covered by the isinstance checks above: a 400-digit int is a valid
        # int and satisfies 0 <= k <= n, but the arithmetic below and in confidence()
        # converts to float, where it raises OverflowError. That is neither ValueError
        # nor TypeError, so it escapes every caller's guard.
        num(k, f"pattern {name!r} k")
        num(n, f"pattern {name!r} n")
        # A quotable number from no observations at all: (0+0.5)/(0+1) == 0.5 makes
        # the consistency check below pass, so it has to be caught here.
        if n == 0 and cell.get("p") is not None:
            raise bad(f"pattern {name!r} quotes p over 0 observations")
        # An interval is a quotable number too: below the floor confidence() falls
        # back to ci[0] as the estimate exactly when n is 0, so a bucket with no
        # observations would report a band from nothing.
        if n == 0 and cell.get("ci") is not None:
            raise bad(f"pattern {name!r} quotes an interval over 0 observations")

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
            expected = round(jeffreys(k, n), 4)
            if abs(p - expected) > 5e-4:
                raise bad(
                    f"pattern {name!r} says p={p} but k/n = {k}/{n} gives {expected}. "
                    f"Refit with python -m chemgraph.tools.ocsr_calibrate "
                    f"instead of editing p."
                )
        check_ci(cell.get("ci"), f"pattern {name!r}")
        label = cell.get("label")
        if label is not None:
            if not isinstance(label, str):
                raise bad(f"pattern {name!r} has a non-string label: {label!r}")
            # Cross-checked against its own p, the way the p is cross-checked
            # against its own k/n. confidence() returns a stored label verbatim
            # above the floor, and an agent is told to band on the label, so a
            # bucket right on 1 of 100 could be labelled unanimous and nothing in
            # the result would contradict it.
            p_value = cell.get("p")
            if isinstance(p_value, (int, float)) and not isinstance(p_value, bool):
                want = _label_for(p_value)
                if label != want:
                    raise bad(f"pattern {name!r} is labelled {label!r} but p="
                              f"{p_value} bands as {want!r}. Refit rather than "
                              f"editing the label.")

    # Checked here so prior_confidence and model_performance can read it without
    # each guarding separately. A string accuracy used to reach _label_for and raise
    # TypeError from inside a lookup, naming neither the table nor the field.
    performance = table.get("model_performance")
    if performance is not None:
        if not isinstance(performance, dict):
            raise bad("'model_performance' must be an object keyed by model name")
        for name, entry in performance.items():
            if not isinstance(entry, dict):
                raise bad(f"model_performance.{name} is not an object")
            # Every field is checked whether or not accuracy is present. Hanging
            # the rest of this block off accuracy left an entry stating p, k, n and
            # ci with no accuracy entirely unchecked, and model_performance() is
            # documented as the one place to ask how good a model is: it handed
            # back a NaN p and a k larger than its own n.
            accuracy = entry.get("accuracy")
            if accuracy is not None:
                if (isinstance(accuracy, bool)
                        or not isinstance(accuracy, (int, float))):
                    raise bad(f"model_performance.{name} has a non-numeric accuracy")
                if not 0.0 <= accuracy <= 1.0:
                    raise bad(f"model_performance.{name} has accuracy={accuracy} "
                              f"outside [0, 1]")
            count = entry.get("n")
            if count is not None:
                if (isinstance(count, bool) or not isinstance(count, int)
                        or count < 0):
                    raise bad(f"model_performance.{name} has an invalid n: {count!r}")
                # Through num() like every other count: it is compared against the
                # floor and divided into k, and nothing else should have to stay
                # bounded for that to be safe.
                num(count, f"model_performance.{name} n")
            # An accuracy backed by nothing is worse than no accuracy: it reads as a
            # real measurement and there is no field left to signal otherwise. A
            # missing n is the same case and looked like the safe one: the floor
            # test in prior_confidence needs an int, so an accuracy with no n skips
            # the floor entirely and a model measured on seven images reports 1.0
            # unanimous, which is exactly what the floor exists to prevent.
            quoted = entry.get("p")
            if (accuracy is not None or quoted is not None) and count is None:
                raise bad(f"model_performance.{name} states a number with no 'n', "
                          f"so no sample floor can apply to it")
            if count == 0 and (accuracy is not None or quoted is not None):
                raise bad(f"model_performance.{name} states a number over 0 "
                          f"observations; drop the entry or set it to null")
            # k reaches the same arithmetic the pattern cells do, so it needs the
            # same guard: a 400-digit int is a valid int and then raises
            # OverflowError from prior_confidence, past every caller's guard.
            hits = entry.get("k")
            if hits is not None:
                if (isinstance(hits, bool) or not isinstance(hits, int)
                        or hits < 0 or (count is not None and hits > count)):
                    raise bad(f"model_performance.{name} has an invalid k: {hits!r}")
                num(hits, f"model_performance.{name} k")
            # 'p' is the field prior_confidence quotes and accuracy is what the
            # listing falls back to, so both owe the cross-check the pattern cells'
            # p owes. k is what makes that check possible, so a stated number
            # without one cannot be checked and is refused rather than trusted.
            if quoted is not None:
                if isinstance(quoted, bool) or not isinstance(quoted, (int, float)):
                    raise bad(f"model_performance.{name} has a non-numeric p")
                if not 0.0 <= quoted <= 1.0:
                    raise bad(f"model_performance.{name} has p={quoted} "
                              f"outside [0, 1]")
            if (accuracy is not None or quoted is not None) and hits is None:
                raise bad(f"model_performance.{name} states a number with no 'k', "
                          f"so nothing can be checked against it")
            if hits is not None and count:
                if accuracy is not None:
                    want = round(hits / count, 4)
                    if abs(accuracy - want) > 5e-4:
                        raise bad(f"model_performance.{name} says accuracy="
                                  f"{accuracy} but k/n = {hits}/{count} gives "
                                  f"{want}. Refit rather than editing accuracy.")
                if quoted is not None:
                    want = round(jeffreys(hits, count), 4)
                    if abs(quoted - want) > 5e-4:
                        raise bad(f"model_performance.{name} says p={quoted} but "
                                  f"Jeffreys on {hits}/{count} gives {want}. Refit "
                                  f"rather than editing p.")
            check_ci(entry.get("ci"), f"model_performance.{name}")
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
    if not isinstance(table, dict):
        return []
    order = _parse_tie_break(table.get("tie_break"))
    return order if order is not None else list(table.get("committee") or [])


def _parse_tie_break(value: object) -> list[str] | None:
    """Model names from a tie_break string, or None when there is nothing to parse.

    An unparseable field returns an empty list, which _validate_calibration rejects
    because it cannot name the committee. A table therefore only reaches
    :func:`tie_break_order` with a usable order or with none recorded at all.
    """
    if value is None:
        return None
    if not isinstance(value, str) or "model-priority:" not in value:
        return []
    _, _, names = value.partition("model-priority:")
    return [n.strip() for n in names.split(",") if n.strip()]


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
    # Public, so it is reachable with a table that never went through the validator.
    # A None here used to raise AttributeError from inside a lookup.
    if not isinstance(table, dict):
        return {"p": None, "label": "unavailable", "n": 0, "ci": None,
                "reason": "no_calibration_table"}

    entry = (table.get("patterns") or {}).get(pattern)
    if entry is None:
        return {"p": None, "label": "unknown", "n": 0, "ci": None,
                "reason": "unknown_pattern"}
    # The table guard above covers the table; the bucket needs its own, for the
    # same reason: unvalidated, a non-object cell raises out of the lookup.
    if not isinstance(entry, dict):
        return {"p": None, "label": "unavailable", "n": 0, "ci": None,
                "reason": "unusable_bucket"}

    p, n, ci = entry.get("p"), entry.get("n", 0), entry.get("ci")
    # The validator rejects a non-integer or negative n, and this function is public,
    # so an unvalidated table can still reach the arithmetic below where n == -1
    # makes the Jeffreys denominator zero.
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return {"p": None, "label": "unavailable", "n": 0, "ci": None,
                "reason": "unusable_bucket"}
    # Honour the floor the table declares, not only the producer's decision to leave
    # p out. A table written by hand, or fitted with a --min-n lower than the one
    # recorded, can carry a point estimate over a handful of images; quoting it
    # would be the false precision the floor exists to prevent.
    # A JSON 20.0 deserializes to float, and bool is an int in Python, so neither
    # isinstance(floor, int) alone nor a bare truth test gets this right.
    floor = table.get("min_n_for_point_estimate")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        floor = None
    if p is not None and floor is not None and n < floor:
        p = None
    if p is None:
        # Below the floor we withhold the number but still owe a useful label, so it
        # is derived from the Jeffreys estimate: the same quantity the `p` column
        # reports above the floor, which keeps one rule across the whole table.
        #
        # The raw point estimate k/n would call a 7/7 bucket "unanimous", claiming
        # certainty from seven items. Jeffreys shrinks toward 0.5 in proportion to
        # how thin the bucket is, putting 7/7 at 0.938 and so at low_n_weak, one
        # band below the 0.95 cut.
        # The stored label is ignored here: a producer writes it beside a point
        # estimate, so it names a full-confidence band. Carrying it through would
        # report "unanimous" from four images, which is the claim the floor exists
        # to refuse, and the low_n_ prefix a caller bands on would be missing.
        k = entry.get("k")
        est = jeffreys(k, n) if (k is not None and n) else (ci[0] if ci else 0.0)
        label = _label_for(est, low_n=True)
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
    # Whichever field is quoted below, not accuracy alone. accuracy is the record of
    # what was counted and nothing requires it, so an entry stating only p reported
    # no prior at all: unavailable from the tool, n=0 for a model measured on 722
    # images, and a listing claiming the table measured it on too few.
    if not entry or (entry.get("p") is None and entry.get("accuracy") is None):
        return {"p": None, "label": "unavailable", "n": 0, "ci": None,
                "reason": "no_prior_for_model"}
    # 'p' is the quotable estimate the fitter wrote, Jeffreys like every pattern
    # cell. 'accuracy' is the raw rate and stays the record of what was counted, so
    # a table without a p (hand-written, or fitted before the field existed) still
    # reports something rather than nothing.
    p = entry["p"] if entry.get("p") is not None else entry["accuracy"]
    n, k = entry.get("n"), entry.get("k")
    # The same floor the agreement buckets are held to, and the same estimator below
    # it. An accuracy is fitted from the same images by the same run, so a table that
    # withholds a bucket's number over three images cannot quote a solo accuracy over
    # those same three, and cannot call it unanimous either: raw k/n is what the
    # Jeffreys rule exists to replace.
    #
    # n is always present alongside an accuracy: the validator rejects an entry
    # without one, because a missing n skips this comparison and reports the floor's
    # opposite. The isinstance guards stay because this function is public and a
    # caller can hand it a dict that never went through the validator.
    floor = t.get("min_n_for_point_estimate")
    if (isinstance(n, int) and not isinstance(n, bool)
            and isinstance(floor, (int, float)) and not isinstance(floor, bool)
            and n < floor):
        est = jeffreys(k, n) if isinstance(k, int) and not isinstance(k, bool) else p
        return {"p": None, "label": _label_for(est, low_n=True), "n": n,
                "ci": entry.get("ci"), "reason": "below_n_floor"}
    return {"p": p, "label": _label_for(p), "n": n if n is not None else 0,
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
        return (
            f"committee_mismatch: the table was fit on {fit} and this machine ran "
            f"{ran}, so no calibrated number applies. Install the rest with: "
            f"pip install 'chemgraph[ocsr]'; or refit for the models you have with "
            f"python -m chemgraph.tools.ocsr_calibrate --labels YOUR_LABELS.csv "
            f"--models {','.join(ran)}"
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
