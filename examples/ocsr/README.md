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

| model | exact match | speed | install |
|-------|-------------|-------|---------|
| `decimer` (default) | 0.899 | 0.7 s | `pip install decimer` |
| `molnextr` | 0.835 | 4.4 s | from source, needs the shim |
| `molscribe` | 0.824 | 5.0 s | from source, needs the shim |
| `ocsrglyph` | 0.766 | 0.3 s | from source |
| `llm` | not measured | varies | none, uses the agent's own model |

Exact match is over a 722-image benchmark. It ranks the four against each other;
it does not predict how any of them will do on your images. DECIMER is the default on both
counts that matter here: most accurate, and the only one that is a plain
`pip install`.

`list_ocsr_models` reports the same table plus what is actually installed on the
machine you are on.

## Installing

```bash
pip install 'chemgraph[ocsr]'
```

That installs DECIMER, and nothing else is needed to use the tool: `model=decimer`
and `model=llm` both work from here.

The other three install from source, and each needs its checkpoint as well, listed
under Checkpoints below. MolNexTR and OCSRGlyph are unpublished on PyPI. MolScribe is
published, but that release pins `torch<2.0` and `numpy<2.0`, which pip would resolve
by downgrading both underneath ChemGraph.

```bash
# OCSRGlyph
git clone https://github.com/EdisonScientific/glyph
pip install --no-deps -e ./glyph

# MolScribe
git clone https://github.com/thomas0809/MolScribe
pip install --no-deps -e ./MolScribe

# MolNexTR ships no package configuration, so it goes on PYTHONPATH
git clone https://github.com/CYF2000127/MolNexTR
export PYTHONPATH="$PWD/MolNexTR:$PYTHONPATH"
pip install pystow matplotlib opencv-python pandas OpenNMT-py==2.2.0 albumentations==1.1.0 SmilesPE
```

Use `--no-deps` for OCSRGlyph and MolScribe. Both pin dependencies that are years
old, and letting pip resolve them downgrades torch and numpy underneath ChemGraph.
Their real requirements are already satisfied by ChemGraph's own.

pip may warn that `molscribe 1.1.1 requires numpy<2.0`. That pin is stale in the
same way its `torch<2.0` is: MolScribe was run over 60 benchmark images on numpy
2.2.6 with no errors and predictions identical to its pinned environment.

All four run in ChemGraph's environment. None needs one of its own, and none
requires changing ChemGraph's pinned versions.

### About the timm shim

MolNexTR and MolScribe each vendor a Swin Transformer written against timm 0.4.12
internals, and pin `timm==0.4.12` for it. Installing that pin breaks OCSRGlyph,
which needs timm 1.x.

`chemgraph.tools.timm_compat` restores the handful of import paths timm moved after
0.4.12, so both models run on current timm with their own source untouched. The
backend applies it before loading either model; there is nothing to configure.
Verified by reproducing MolNexTR's timm 0.4.12 predictions exactly across 60
benchmark images.

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

Each is on Hugging Face; the paths above are what `ocsr_registry.json` records, and
pointing `CHEMGRAPH_OCSR_REGISTRY` at your own copy of that file moves them.

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
