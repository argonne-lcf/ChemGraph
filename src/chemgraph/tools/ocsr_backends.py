"""Backends that actually run a model, for the OCSR tool.

Split from :mod:`chemgraph.tools.ocsr_core` so that core stays pure: core imports and
its tests run on a host with no network, no credentials and no conda environments.
Everything here touches a socket or a subprocess.

Two families:

* :func:`smiles_from_llm` posts a base64 image to an OpenAI-compatible vision endpoint
  (ALCF, or a local Argo shim).
* :func:`smiles_from_specialist` drives one of the purpose-built OCSR networks
  (MolNexTR, MolScribe, DECIMER, OCSRGlyph) as a subprocess.

Both return the same narrow dict and **never raise**::

    {"ok", "smiles", "raw", "model_used", "cold_start", "latency_s", "error"}

``ocsr_tools`` maps that onto the public 18-key contract. ``raw`` is dropped there
rather than returned, since ``build_result`` rejects keys outside the contract.
"""

from __future__ import annotations

import logging
import os
import time

from chemgraph.tools import ocsr_core as core
from chemgraph.tools import ocsr_models as models

logger = logging.getLogger(__name__)

ALCF_BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
SHIM_BASE_URL = "http://127.0.0.1:11216/argoapi/v1"

# Phrases that mark an HTTP 200 body as a gateway notice rather than a model answer.
# The Argo shim returns ACCESS DENIED with a 200 when the `user` field is a placeholder,
# and an OCSR answer is a short SMILES with no prose, so any of these means failure.
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


def _llm_credentials(backend: str) -> tuple[str, str, dict | None]:
    """Resolve (base_url, api_key, extra_headers) for one endpoint.

    Per-backend rather than a chained ``or``: with both ALCF_ACCESS_TOKEN and
    OPENAI_API_KEY set, which is common, a fallback chain would make one endpoint
    unreachable.
    """
    if backend == "alcf":
        key = os.environ.get("ALCF_ACCESS_TOKEN")
        if not key:
            raise RuntimeError(
                "ALCF_ACCESS_TOKEN is not set. Mint one with "
                "`python inference_auth_token.py get_access_token`."
            )
        return ALCF_BASE_URL, key, None

    if backend == "shim":
        key = os.environ.get("ARGO_SHIM_API_KEY")
        if not key:
            raise RuntimeError(
                "ARGO_SHIM_API_KEY is not set. The Argo shim gates on an x-api-key "
                "header; the key rotates when the shim restarts."
            )
        # openai>=2 also reads OPENAI_CUSTOM_HEADERS, but pass it explicitly so the
        # call does not depend on the caller's environment being set up first.
        return SHIM_BASE_URL, key, {"x-api-key": key}

    raise RuntimeError(f"{backend!r} is not an LLM backend")


def _pick_reachable_llm() -> str:
    """For backend='llm': shim if its key is present, else ALCF.

    Deliberately checks credentials rather than probing the network. A probe costs a
    round trip on every call and would still race with an endpoint going cold.
    """
    if os.environ.get("ARGO_SHIM_API_KEY"):
        return "shim"
    return "alcf"


def smiles_from_llm(image_bytes: bytes, mime: str, backend: str,
                    model: str | None = None, timeout_s: float = 90.0) -> dict:
    """Read a structure image with a vision LLM.

    Parameters
    ----------
    image_bytes, mime
        From :func:`chemgraph.tools.ocsr_core.load_image_bytes`, so the bytes were
        validated before anything was sent anywhere.
    backend
        ``"alcf"``, ``"shim"``, or ``"llm"`` to pick whichever is configured.
    model
        Endpoint-specific name. ``None`` takes that endpoint's default. The two
        endpoints use different formats and :func:`ocsr_models.resolve_model`
        translates, so a caller can pass the friendly ``argo:claude-opus-4.8``.
    """
    import base64

    start = time.monotonic()
    if backend == "llm":
        backend = _pick_reachable_llm()

    try:
        base_url, key, headers = _llm_credentials(backend)
        wire = models.resolve_model(backend, model)
    except (RuntimeError, ValueError) as e:
        return _narrow(error=str(e), latency_s=time.monotonic() - start)

    try:
        from openai import OpenAI

        from chemgraph.prompt.ocsr_prompt import OCSR_SYSTEM_PROMPT, OCSR_USER_PROMPT

        client = OpenAI(base_url=base_url, api_key=key, timeout=timeout_s,
                        max_retries=0, default_headers=headers)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = client.chat.completions.create(
            model=wire,
            messages=[
                {"role": "system", "content": OCSR_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": OCSR_USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]},
            ],
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "not ready" in msg or "internal_endpoint_error" in msg:
            msg = (f"the {backend} endpoint for {wire} is cold (503). It says nothing "
                   f"about the model; try again shortly or pick another model.")
        return _narrow(model_used=wire, error=msg[:400],
                       latency_s=time.monotonic() - start)

    elapsed = time.monotonic() - start

    if any(m in raw.lower() for m in _ERROR_MASQUERADE):
        return _narrow(raw=raw, model_used=wire, latency_s=elapsed,
                       error=f"the endpoint returned an auth notice, not an answer: {raw[:160]}")

    smiles = core.extract_smiles(raw)
    if smiles is None:
        return _narrow(raw=raw, model_used=wire, latency_s=elapsed,
                       error="the model replied but no SMILES could be extracted")
    return _narrow(ok=True, smiles=smiles, raw=raw, model_used=wire, latency_s=elapsed)


# ---------------------------------------------------------------------------
# Local specialist models
# ---------------------------------------------------------------------------


_client = None


def _specialist_client():
    """The in-tree worker client, or None when nothing is installed here.

    Uses :mod:`chemgraph.tools.ocsr_worker_client`, which ships with ChemGraph: no
    private repo, no ``sys.path`` surgery, no external checkout. It owns worker
    lifetime, the idle reaper and the temp-image handling, so this function is only
    a lazily-built singleton.
    """
    global _client
    if _client is not None:
        return _client
    try:
        from chemgraph.tools.ocsr_setup import load_install
        from chemgraph.tools.ocsr_worker_client import OCSRWorkerClient

        config = load_install()
        if not config:
            return None
        _client = OCSRWorkerClient(config)
        return _client
    except BaseException as e:  # never let a config problem kill the caller
        logger.warning("specialist client unavailable: %s: %s", type(e).__name__, e)
        return None


def available_specialists() -> list[str]:
    """Names whose conda env exists on this host. Cheap: stat only, never spawns."""
    try:
        from chemgraph.tools.ocsr_setup import load_install
    except Exception:
        return []
    out = []
    for name, entry in (load_install() or {}).items():
        py = os.path.expandvars(os.path.expanduser(entry.get("python_bin", "")))
        if py and os.path.exists(py):
            out.append(name)
    return out


def _setup_hint() -> str:
    """Tell the caller how to fix a missing install, in a command that actually runs.

    Built at call time from the registry: the model name is whichever specialist the
    registry marks as the default, and ocsr_setup takes it as a bare positional. A
    hardcoded string got both wrong at once, naming an `install` subcommand that does
    not exist and a model that a custom registry need not contain. This message is
    returned into an agent's context, so a command that exits 2 costs a retry loop.
    """
    from chemgraph.tools.ocsr_models import DEFAULT_SPECIALIST

    return (
        "No local OCSR models are installed. Install one with "
        f"`python -m chemgraph.tools.ocsr_setup {DEFAULT_SPECIALIST}`, or use a "
        "vision LLM with backend='alcf'."
    )


def smiles_from_specialist(name: str, image_bytes: bytes,
                           timeout_s: float | None = None) -> dict:
    """Read a structure image with one local specialist model.

    The first call for a given model loads it: 5-25 s for most, 50-66 s for DECIMER,
    measured on a busy shared machine; later calls are 0.3-5 s. ``cold_start`` in the
    return says which happened, so an agent can reason about cost instead of guessing.
    """
    start = time.monotonic()
    bare = name.removeprefix("local:")

    installed = available_specialists()
    if bare not in installed:
        # Name the command for the model that is missing. Appending the generic hint
        # unconditionally asserted "no local models are installed" right after
        # listing two of them, and pointed at a model the user already had.
        if installed:
            detail = (f"{bare!r} is not installed. Installed: "
                      f"{', '.join(installed)}. Install it with: "
                      f"python -m chemgraph.tools.ocsr_setup {bare}")
        else:
            detail = _setup_hint()
        return _narrow(model_used=bare, cold_start=True, error=detail,
                       latency_s=time.monotonic() - start)

    client = _specialist_client()
    if client is None:
        return _narrow(model_used=bare, cold_start=True, error=_setup_hint(),
                       latency_s=time.monotonic() - start)

    # predict() never raises and already reports cold_start and infer_s.
    r = client.predict(bare, image_bytes, timeout_s=timeout_s)
    cold = r.get("cold_start", False)
    elapsed = r.get("infer_s", time.monotonic() - start)

    if not r["ok"]:
        return _narrow(model_used=bare, cold_start=cold, latency_s=elapsed,
                       error=r.get("error", "the model returned no SMILES"))

    # The worker protocol sets ok = bool(smiles): it means "the model returned a
    # string", not "the string is a molecule". 3.8% of real specialist predictions do
    # not parse on average and 5.4% for the worst of the four, and a further class is
    # prose. Gate on extract_smiles, the same check the LLM path uses: it rejects both
    # kinds, and rejects a refusal like
    # "I cannot process images" that RDKit would otherwise read as iodine.
    #
    # Without this the caller gets ok=True with the model's 0.899 prior attached to
    # a string that is not a molecule, and formula=None / n_fragments=0 means the
    # multi-fragment guard cannot fire either.
    smiles = core.extract_smiles(r.get("smiles"))
    if smiles is None:
        return _narrow(
            model_used=bare, cold_start=cold, latency_s=elapsed,
            raw=str(r.get("smiles") or "")[:200],
            error=(f"{bare} returned a string RDKit cannot read as a molecule: "
                   f"{str(r.get('smiles'))[:80]!r}"),
        )

    return _narrow(ok=True, smiles=smiles, model_used=bare,
                   cold_start=cold, latency_s=elapsed)


def _reset_client() -> None:
    """Drop the singleton so a test can force a fresh one."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None


