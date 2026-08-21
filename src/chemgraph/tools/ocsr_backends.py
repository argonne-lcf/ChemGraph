"""Backends that actually run a model, for the OCSR tool.

Split from :mod:`chemgraph.tools.ocsr_core` so core stays pure: core imports and its
tests run on a host with no models and no network. Everything here loads a checkpoint
or opens a socket.

Two families:

* :func:`smiles_from_llm` posts a base64 image to a vision LLM. The agent passes its
  own bound model in, so the tool never resolves credentials itself.
* :func:`smiles_from_specialist` runs one of the four purpose-built OCSR networks
  (DECIMER, MolNexTR, MolScribe, OCSRGlyph) in this process.

All four specialists install into ChemGraph's own environment. MolNexTR and MolScribe
each vendor a Swin implementation written against timm 0.4.12 internals, so they are
imported after :mod:`chemgraph.tools.timm_compat` restores those paths on timm 1.x.
Nothing here spawns a subprocess or needs a per-model environment.

Both families return the same narrow dict and **never raise**::

    {"ok", "smiles", "raw", "model_used", "cold_start", "latency_s", "error"}

``ocsr_tools`` maps that onto the public result contract. ``raw`` is dropped there,
since :func:`~chemgraph.tools.ocsr_core.build_result` rejects keys outside it.

Models are loaded once and cached: the first call pays to build the network and read
the checkpoint (9-170 s measured cold on a shared CPU node), later calls are 0.3-5 s.
``cold_start`` in the return says which happened, so an agent can reason about the
cost instead of guessing.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from typing import Any, Callable

from chemgraph.tools import ocsr_core as core
from chemgraph.tools import ocsr_models as models

logger = logging.getLogger(__name__)

# Phrases that mark an HTTP 200 body as a gateway notice instead of a model answer.
# An OCSR answer is a short SMILES with no prose, so any of these means failure.
_ERROR_MASQUERADE = (
    "access denied", "not authorized", "unauthorized",
    "invalid or missing", "api key", "authentication",
)


def _narrow(ok=False, smiles=None, raw="", model_used=None,
            cold_start=False, latency_s=0.0, error="") -> dict:
    """The shape both backends return. Keeps the two paths honest with each other."""
    return {"ok": ok, "smiles": smiles, "raw": raw, "model_used": model_used,
            "cold_start": cold_start, "latency_s": round(latency_s, 3), "error": error}


# ---------------------------------------------------------------------------
# Vision LLM
# ---------------------------------------------------------------------------


def smiles_from_llm(image_bytes: bytes, mime: str, llm: Any,
                    structured: bool = False) -> dict:
    """Read a structure image with the agent's own LLM.

    Parameters
    ----------
    image_bytes, mime
        From :func:`chemgraph.tools.ocsr_core.load_image_bytes`, so the bytes were
        sniffed and size-checked before anything is sent anywhere.
    llm
        The LangChain chat model the agent is already using. Passing it in is what
        keeps this path from reading credentials out of the environment: ChemGraph
        cannot yet give a sub-agent a different model, so the fallback is by
        definition the agent's own.
    structured
        Ask for ``{"smiles": ...}`` instead of a bare string. The reply is read the
        same way either way; this only makes the named-field form more likely, and
        gives a non-molecule image an explicit null to answer with.

    A model without vision support fails here as an API error, and is reported like
    any other failure. Pre-empting it would mean keeping a list of vision-capable
    models, which changes faster than this module could track.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from chemgraph.prompt import ocsr_prompt

    start = time.monotonic()
    if llm is None:
        return _narrow(error="no LLM was bound to this tool, so the fallback is "
                             "unavailable. Install a specialist model, or pass "
                             "model= one of them.")

    name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "llm"
    b64 = base64.b64encode(image_bytes).decode("ascii")

    try:
        resp = llm.invoke([
            SystemMessage(content=ocsr_prompt.OCSR_STRUCTURED_SYSTEM_PROMPT
                          if structured else ocsr_prompt.OCSR_SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": ocsr_prompt.OCSR_USER_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]),
        ])
        raw = (resp.content if isinstance(resp.content, str)
               else str(resp.content)).strip()
    except Exception as e:
        return _narrow(model_used=name, latency_s=time.monotonic() - start,
                       error=f"{type(e).__name__}: {e}"[:400])

    elapsed = time.monotonic() - start

    if any(m in raw.lower() for m in _ERROR_MASQUERADE):
        return _narrow(raw=raw, model_used=name, latency_s=elapsed,
                       error=f"the endpoint returned an auth notice instead of "
                             f"an answer: {raw[:160]}")

    smiles = core.extract_smiles(raw)
    if smiles is None:
        return _narrow(raw=raw, model_used=name, latency_s=elapsed,
                       error="the model replied but no SMILES could be extracted")
    return _narrow(ok=True, smiles=smiles, raw=raw, model_used=name,
                   latency_s=elapsed)


# ---------------------------------------------------------------------------
# Local specialists
# ---------------------------------------------------------------------------

# One loaded model per name. Building the network and reading the checkpoint costs
# far more than inference, so a session that reads twenty images pays it once.
_LOADED: dict[str, Callable[[str], Any]] = {}
_LOAD_LOCK = threading.Lock()


def _device() -> str:
    """Pick a device string. CPU unless a working accelerator is present.

    XPU is resolved but untested: this has only run on CPU so far, and the check
    stays cheap enough to leave in.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        try:
            if xpu.is_available():
                return "xpu"
        except Exception:
            pass
    return "cpu"


def _load_decimer(_weights: str | None) -> Callable[[str], Any]:
    """DECIMER is TensorFlow-based and manages its own checkpoint cache.

    Weights land in ``~/.data/DECIMER-V2`` through pystow on first use and are reused
    from there afterwards, so there is no checkpoint path to pass.
    """
    from DECIMER import predict_SMILES

    return predict_SMILES


def _load_ocsrglyph(weights: str) -> Callable[[str], Any]:
    from glyph.ocsr.predict import OCSRPredictor

    # The constructor wants the device as a plain string, not a torch.device.
    model = OCSRPredictor(weights, device=_device())
    return model.predict


def _load_molnextr(weights: str) -> Callable[[str], Any]:
    from chemgraph.tools import timm_compat

    timm_compat.install()
    from MolNexTR.model import molnextr

    model = molnextr(weights, device=_device())

    def infer(image_path: str):
        return model.predict_final_results(image_path)

    return infer


def _load_molscribe(weights: str) -> Callable[[str], Any]:
    from chemgraph.tools import timm_compat

    timm_compat.install()
    import torch
    from molscribe import MolScribe

    model = MolScribe(weights, device=torch.device(_device()))
    return model.predict_image_file


_LOADERS: dict[str, Callable[[str], Callable[[str], Any]]] = {
    "decimer": _load_decimer,
    "ocsrglyph": _load_ocsrglyph,
    "molnextr": _load_molnextr,
    "molscribe": _load_molscribe,
}

# Where each model's prediction hides in what it returns. DECIMER and OCSRGlyph hand
# back a bare string; the other two return a dict under differing keys.
_SMILES_KEYS = ("predicted_smiles", "smiles", "pred_smiles", "SMILES")


def _unwrap(result: Any) -> str | None:
    """Pull the SMILES string out of whatever shape a model returned."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in _SMILES_KEYS:
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(result, (list, tuple)) and result:
        return _unwrap(result[0])
    return None


def is_installed(name: str) -> bool:
    """Whether ``name``'s package can be imported, without loading the model."""
    import importlib.util

    spec = models.SPECIALIST_MODELS.get(name)
    module = (spec or {}).get("import_name")
    if not module:
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_specialists() -> list[str]:
    """Specialists whose package is importable here, in registry order."""
    return [name for name in models.SPECIALIST_MODELS if is_installed(name)]


# Where checkpoints live, overriding the registry defaults. On a cluster the weights
# are usually on shared scratch, not under $HOME, and moving one file should not mean
# copying the whole registry.
_WEIGHTS_DIR_ENV = "CHEMGRAPH_OCSR_WEIGHTS_DIR"


def checkpoint_path(name: str) -> str | None:
    """The resolved checkpoint path for ``name``, or None if it needs none.

    DECIMER fetches and caches its own weights, so it has no entry.
    """
    spec = models.SPECIALIST_MODELS.get(name) or {}
    weights = (spec.get("install") or {}).get("weights")
    if not weights:
        return None
    root = os.environ.get(_WEIGHTS_DIR_ENV)
    if root:
        return os.path.join(os.path.expanduser(root), name,
                            os.path.basename(weights))
    return os.path.expanduser(weights)


def _preload_torchvision() -> None:
    """Import torchvision before anything can pull in TensorFlow.

    Loading torchvision into a process that already holds TensorFlow segfaults inside
    ``torchvision.ops``, so reading one image with DECIMER and the next with any torch
    model kills the process. The reverse order is safe, and importing torch alone is
    safe; torchvision is the one that has to go first.

    Called before loading any specialist, so importing this module on an install
    without torch stays free: ``llm_agent`` reaches here
    through ``ocsr_agent`` on every workflow, including the ones that use no OCSR
    model at all.
    """
    try:
        import torchvision  # noqa: F401
    except ImportError:  # pragma: no cover - torch is optional until a model is used
        pass


def _install_hint() -> str:
    """How to install a missing model. One extra covers all four."""
    return (f"Install it with: pip install 'chemgraph[ocsr]'. "
            f"Installed here: {', '.join(available_specialists()) or 'none'}.")


def _resolve_weights(name: str) -> tuple[str | None, str]:
    """Return (path, error). A missing checkpoint is an error naming the fix."""
    path = checkpoint_path(name)
    if path is None:
        return None, ""
    if os.path.exists(path):
        return path, ""
    return None, (f"{name}'s checkpoint is missing at {path}. "
                  f"examples/ocsr/README.md lists where to download it, and "
                  f"CHEMGRAPH_OCSR_WEIGHTS_DIR moves where it is looked for.")


def _get_model(name: str) -> tuple[Callable[[str], Any] | None, bool, str]:
    """Return (callable, cold_start, error), loading and caching on first use."""
    cached = _LOADED.get(name)
    if cached is not None:
        return cached, False, ""

    with _LOAD_LOCK:
        # Another thread may have loaded it while this one waited.
        cached = _LOADED.get(name)
        if cached is not None:
            return cached, False, ""

        weights, error = _resolve_weights(name)
        if error:
            return None, True, error

        _preload_torchvision()
        try:
            loaded = _LOADERS[name](weights)
        except ImportError as e:
            return None, True, (f"{name} is not installed ({e}). "
                                f"{_install_hint()}")
        except Exception as e:
            return None, True, f"loading {name} failed: {type(e).__name__}: {e}"[:400]

        _LOADED[name] = loaded
        return loaded, True, ""


def smiles_from_specialist(name: str, image_path: str) -> dict:
    """Read a structure image with one local specialist model.

    Takes a path, since every one of the four reads the file itself and handing
    them bytes would mean writing a temporary copy just to hand it back.
    """
    start = time.monotonic()
    bare = name.removeprefix("local:")

    if bare not in _LOADERS:
        return _narrow(model_used=bare, latency_s=time.monotonic() - start,
                       error=f"unknown model {bare!r}. Choose one of: "
                             f"{', '.join(_LOADERS)}, or 'llm'.")

    model, cold, error = _get_model(bare)
    if model is None:
        return _narrow(model_used=bare, cold_start=cold, error=error,
                       latency_s=time.monotonic() - start)

    try:
        raw_result = model(image_path)
    except Exception as e:
        return _narrow(model_used=bare, cold_start=cold,
                       latency_s=time.monotonic() - start,
                       error=f"{bare} failed on this image: "
                             f"{type(e).__name__}: {e}"[:400])

    elapsed = time.monotonic() - start
    raw = _unwrap(raw_result)

    # A model returning a string means "it produced output", not "the output is a
    # molecule". Around 4% of real specialist predictions do not parse, and a further
    # class is prose. Gate on extract_smiles, the same check the LLM path uses, so a
    # caller never gets ok=True with something RDKit cannot read.
    smiles = core.extract_smiles(raw)
    if smiles is None:
        return _narrow(model_used=bare, cold_start=cold, latency_s=elapsed,
                       raw=str(raw or "")[:200],
                       error=f"{bare} returned a string RDKit cannot read as a "
                             f"molecule: {str(raw)[:80]!r}")

    return _narrow(ok=True, smiles=smiles, raw=str(raw or "")[:200],
                   model_used=bare, cold_start=cold, latency_s=elapsed)

