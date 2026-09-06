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
structure it was drawn from. `--ensemble` votes every installed specialist and
prints the agreement and confidence beside each result. `--agent` routes the same
call through an LLM, and `--llm` picks which one.

## Models

`model=` takes one of four specialists or `llm`:

| model | exact match | speed |
|-------|-------------|-------|
| `decimer` (default) | 0.898 | 0.7 s |
| `molnextr` | 0.835 | 4.4 s |
| `molscribe` | 0.824 | 5.0 s |
| `ocsrglyph` | 0.766 | 0.3 s |
| `llm` | not measured | varies |

Each figure is the Jeffreys estimate `(k+0.5)/(n+1)` over the 722-image benchmark,
which is what `list_ocsr_models` prints and what a single-model read bands on. The
raw counts are in the calibration table's `model_performance`.

`pip install 'chemgraph[ocsr]'` installs all four specialists. `llm` needs no
install and uses the agent's own model.

The specialists return a SMILES and nothing else. An LLM returns whatever it likes,
so the tool pulls the SMILES back out of markdown, parentheses, code fences and
prose. `structured=True` asks the model for `{"smiles": ...}` instead, which puts
the answer in a named field and lets an image that is not a molecule reply with
null instead of a guess:

```python
image_to_smiles_core("diagram.png", model="llm", structured=True)
```

It changes only the system prompt sent to the model, so the four specialists ignore
it. The published accuracies were measured under the other prompt.

Exact match is over a 722-image benchmark. It ranks the four against each other;
it does not predict how any of them will do on your images. DECIMER is the default because it is the most accurate of the four.

`list_ocsr_models` reports what is actually installed on the machine you are on.
The accuracies it prints come from the calibration table, so they follow a refit
(see below); the numbers above are the shipped table's, measured on the benchmark.

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
    "confidence": None,      # single model: no per-image number, see below
    "confidence_interval": None,  # the 95% interval, when a committee measured one
    "confidence_label": "weak",   # bands this model's measured solo accuracy
    "confidence_unavailable_reason": "single_model_has_no_per_image_confidence",
    "agreement": None,       # how a committee split, when one ran
    "votes": None,           # which models produced which SMILES
    "abstained": None,       # models that ran and returned nothing usable
    "backend_used": "specialist",   # or "llm", or "ensemble"
    "model_used": "decimer",
    "cold_start": False,
    "latency_s": 0.7,
    "error": "",
    "warning": "",
}
```

A single model reports no confidence, because it cannot say how likely it is to be
right about one particular image, and quoting its benchmark accuracy would answer a
question about someone else's images.

## Confidence from a committee

`ensemble=True` reads the image with every installed specialist and votes. How much
they agree is measurable, and it was measured on 722 benchmark images:

| agreement | meaning | P(correct) | n |
|-----------|---------|-----------:|---:|
| `4` | all four agree | 0.9989 | 462 |
| `3/1` | three against one | 0.9816 | 135 |
| `2/1/1` | two agree, two differ | 0.8017 | 57 |
| `2/2` | an even split | below the sample floor | 12 |
| `1/1/1/1` | all different | 0.3772 | 56 |

The strongest single model is right 89.8% of the time, so a unanimous committee
turns a one-in-ten error rate into one in a thousand, and an all-different vote is
worse than a coin flip. A model that ran and returned nothing usable counts as a
dissenting singleton, so the parts always sum to the committee size.

```python
result = image_to_smiles_core("molecule.png", ensemble=True)
# {'ok': True, 'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'confidence': 0.9989,
#  'confidence_interval': [0.9946, 1.0], 'confidence_label': 'unanimous',
#  'agreement': '4',
#  'votes': {'CC(=O)Oc1ccccc1C(=O)O': ['decimer', 'molnextr', ...]}, ...}
```

Buckets below 20 observations get a label and an interval but no point estimate: at
that size the 95% interval is 41 points wide, so a decimal would be false precision.
`confidence_interval` carries it either way, and is the only quantitative thing a
thin bucket can honestly offer.

The number scores the skeleton only: `"scoring": "stereo_blind"` in the table, so
the committee is counted after stereochemistry is stripped. Models that agree on
the skeleton can still have read different wedge bonds, and the answer then takes
the reading of the model highest in the table's `tie_break` order. That order ranks
by overall accuracy, which is close to but not the same as accuracy on
stereochemistry, so the answer is not always the best-placed reading of the wedge
bonds. `warning` says when a reading was overruled, and it can accompany a
confidence of 0.9989.

`confidence_label` is one of `unanimous` (p >= 0.99), `strong` (>= 0.95), `weak`
(>= 0.70), or `conflicting` below that, prefixed `low_n_` when the bucket is thin.
`unknown` means the table has no bucket for this split, and `unavailable` that no
number applies at all; `confidence_unavailable_reason` says which case it is. On a
single-model call the label bands that model's measured solo accuracy, since there
is no per-image number to band.

### Refitting for your own images

The shipped table describes those four models on RDKit-rendered diagrams. Scans,
photographs and journal crops have a different relationship between agreement and
correctness, so fit your own:

```bash
python -m chemgraph.tools.ocsr_calibrate --labels labels.csv --out mine.json
export CHEMGRAPH_OCSR_CALIBRATION=mine.json
```

`--models` fits a subset of what is installed, `--tie-break` overrides the priority
the fitter derives from your own measurements, and `--min-n` moves the floor below
which a bucket gets a label but no number. `image_to_smiles_core` also takes
`calibration=` for one call, where the environment variable covers a session.

`labels.csv` is `image_path,smiles` with a reference SMILES per image. If you have
no labels to spare, `--validate` scores the current table against a sample instead,
which takes far fewer of them than fitting a new one. It checks whatever the tool
would use: `CHEMGRAPH_OCSR_CALIBRATION` when that is set, and the packaged table
otherwise.

When the table covers fewer models than are installed, pass
`models_wanted=["decimer", "molnextr"]` so the committee that runs is the one the
table describes. Otherwise every installed model votes and no calibrated number
applies.

## Images

Four structures, drawn by RDKit from known SMILES so the answers can be checked:
`aspirin`, `caffeine`, `imatinib`, and `penicillin_g` (which has stereocentres, where
the models disagree most).
