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

import logging
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
# longest in the OCSR benchmark is 224.
_MAX_SMILES_CHARS = 4000

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


# ---------------------------------------------------------------------------
# SMILES handling
# ---------------------------------------------------------------------------


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


def build_result(**overrides) -> dict:
    """Assemble the tool's return dict, with every contract key present.

    One place builds this shape, so a backend cannot ship a partial dict and an agent
    never has to test for a missing key. The docstring of ``image_to_smiles`` promises
    these exact fields; changing the set means changing both together.

    The committee keys (agreement, votes, abstained) belong to the ensemble work and
    are not part of this contract; a single model has nothing to report for them.
    """
    result = {
        "ok": False,
        "smiles": None,
        "valid": False,
        "formula": None,
        "n_fragments": 0,
        "model_used": None,
        "cold_start": False,
        "latency_s": 0.0,
        "error": "",
        "warning": "",
    }
    unknown = set(overrides) - set(result)
    if unknown:
        raise KeyError(f"not part of the OCSR result contract: {sorted(unknown)}")
    result.update(overrides)
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
