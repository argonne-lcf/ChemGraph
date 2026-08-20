# Image to SMILES

Read a molecule's 2D structure diagram and get its SMILES back.

```python
from chemgraph.agent.llm_agent import ChemGraph

agent = ChemGraph(model_name="gpt-4o", workflow_type="ocsr")
agent.run("What molecule is in examples/ocsr/images/aspirin.png?")
```

Or without an agent:

```python
from chemgraph.tools.ocsr_tools import image_to_smiles_core

image_to_smiles_core("examples/ocsr/images/aspirin.png")
# {'ok': True, 'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'valid': True,
#  'formula': 'C9H8O4', 'model_used': 'decimer', ...}
```

`python run_ocsr.py` runs all four images and checks each answer against the
structure it was drawn from. `--agent` routes the same call through an LLM.

## Models

`model=` takes one of four specialists or `llm`:

| model | exact match | speed |
|-------|-------------|-------|
| `decimer` (default) | 0.899 | 0.7 s |
| `molnextr` | 0.835 | 4.4 s |
| `molscribe` | 0.824 | 5.0 s |
| `ocsrglyph` | 0.766 | 0.3 s |
| `llm` | not measured | varies |

`pip install 'chemgraph[ocsr]'` installs all four specialists. `llm` needs no
install and uses the agent's own model.

Exact match is over a 722-image benchmark. It ranks the four against each other;
it does not predict how any of them will do on your images. DECIMER is the default because it is the most accurate of the four.

`list_ocsr_models` reports the same table plus what is actually installed on the
machine you are on.

## Installing

```bash
pip install 'chemgraph[ocsr]'
```

That installs all four specialists. Three of them also need a checkpoint on disk,
listed under Checkpoints below; DECIMER fetches its own.

DECIMER is on PyPI. The other three install from GitHub, pinned to a commit so the
extra keeps resolving to what was tested here. MolNexTR and MolScribe point at forks
whose only change is two lines of `setup.py` each: both pin `timm==0.4.12`, OCSRGlyph
needs timm 1.x, and no single version satisfies both.

Installing them one at a time lets the later install replace the timm the earlier one
needs, and pip reports success while OCSRGlyph fails at inference with
`RuntimeError: features_only not implemented for Vision Transformer models`.

MolScribe is on PyPI, but that release pins `torch>=1.11.0,<2.0`, which pip resolves
by downgrading torch underneath ChemGraph, so the extra installs from git. Current
git has relaxed the torch pin and keeps `numpy>=1.19.5,<2.0`, which the fork relaxes
too: MolScribe was run over the benchmark images on numpy 2.2.6 with predictions
identical to its pinned environment.

All four run in ChemGraph's environment. None needs one of its own, and none requires
changing ChemGraph's pinned versions.

### About the timm shim

MolNexTR and MolScribe each vendor a Swin Transformer written against timm 0.4.12
internals. `chemgraph.tools.timm_compat` restores the handful of import paths timm
moved after 0.4.12, so both models run on current timm with their own source
untouched. The backend applies it before loading either model; there is nothing to
configure. Verified by reproducing MolNexTR's timm 0.4.12 predictions exactly across
the benchmark images.

## Checkpoints

DECIMER downloads its weights on first use and caches them in `~/.data/DECIMER-V2`
through pystow, so it manages itself.

The other three need a checkpoint on disk before they can run. The tool checks for
it and tells you what is missing instead of failing inside the model:

| model | default location | size |
|-------|------------------|------|
| `molnextr` | `~/ocsr-weights/molnextr/molnextr_best.pth` | 1.1 GB |
| `molscribe` | `~/ocsr-weights/molscribe/swin_base_char_aux_1m.pth` | 1.1 GB |
| `ocsrglyph` | `~/ocsr-weights/ocsrglyph/model.pth` | 0.4 GB |

All three are on Hugging Face. MolNexTR publishes its checkpoint in a dataset repo,
hence the `--repo-type`:

```bash
pip install huggingface_hub
hf download CYF200127/MolNexTR molnextr_best.pth --repo-type dataset \
    --local-dir ~/ocsr-weights/molnextr
hf download yujieq/MolScribe swin_base_char_aux_1m.pth \
    --local-dir ~/ocsr-weights/molscribe
hf download EdisonScientific/OCSRGlyph model.pth \
    --local-dir ~/ocsr-weights/ocsrglyph
```

The paths above are what `ocsr_registry.json` records. `CHEMGRAPH_OCSR_WEIGHTS_DIR`
moves the parent directory, and pointing `CHEMGRAPH_OCSR_REGISTRY` at your own copy
of that file moves each path individually.

A model is loaded once per process. The first call pays for building the network and
reading the checkpoint, measured cold on a shared CPU node at 9 s for MolScribe, 19 s
for MolNexTR, 75 s for OCSRGlyph and 168 s for DECIMER. Later calls in the same
process are the per-image times in the table above. `cold_start` in the result says
which happened, and `latency_s` includes the load when it did.

## Reading the result

```python
{
    "ok": True,              # a parseable SMILES came back
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "valid": True,           # RDKit parsed it
    "formula": "C9H8O4",
    "n_fragments": 1,        # above 1 means a salt, mixture, or reaction scheme
    "model_used": "decimer",
    "cold_start": False,
    "latency_s": 0.7,
    "error": "",
    "warning": "",
}
```

No confidence number is reported. A single model cannot say how likely it is to be
right about one particular image, and quoting its benchmark accuracy here would
answer a question about someone else's images. If an answer matters, run a second
model and see whether the two agree.

## Images

Four structures, drawn by RDKit from known SMILES so the answers can be checked:
`aspirin`, `caffeine`, `imatinib`, and `penicillin_g` (which has stereocentres, where
the models disagree most).
