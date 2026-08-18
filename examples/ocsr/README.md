# Reading structure diagrams with ChemGraph

Optical Chemical Structure Recognition: give the **`image_to_smiles`** tool a picture
of a molecule's 2D structure and get a SMILES back, with a calibrated confidence.

The tool calls the recognition model itself, so the agent never sees the image. That
means it works with any LLM, including ones with no vision capability at all.

## What's here
- `run_chemgraph.py`: reads the sample image, first as a plain function call, then
  through a ChemGraph agent with `--agent`.
- `aspirin.png`: a rendered structure diagram to try it on.

## Setup

Nothing is required for the vision-LLM backends beyond a token:

```bash
export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)
```

The local specialist models are optional and usually more accurate on ordinary
structure diagrams. They are purpose-built
image-to-SMILES networks, each needing its own conda environment, so they are not pip
dependencies. Install at least DECIMER to use the default backend:

```bash
python -m chemgraph.tools.ocsr_setup --list      # sizes and what is already there
python -m chemgraph.tools.ocsr_setup decimer     # ~3.3 GB; the default backend
# the others only matter for backend="ensemble":
python -m chemgraph.tools.ocsr_setup molnextr
```

Go through `ocsr_setup` rather than running the build scripts by hand. It finds the
script inside the installed package, where you have no reason to know the path, and
it records where the model landed in `~/.chemgraph/ocsr_install.json`. A model built
by invoking the script directly is not recorded, so the tool will not find it.

Check what is available on this machine:

```bash
python -m chemgraph.tools.ocsr_models
```

## Run

```bash
python run_chemgraph.py             # direct calls, no LLM needed

# the agent path needs a model that can call tools; pick any ChemGraph supports
export OPENAI_API_KEY=...           # or, on ALCF:
export CHEMGRAPH_MODEL=google/gemma-4-31B-it
export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)
python run_chemgraph.py --agent
```

The `ocsr` workflow binds these two tools and nothing else:

```python
import asyncio
from chemgraph.agent.llm_agent import ChemGraph

cg = ChemGraph(model_name="gpt-4o", workflow_type="ocsr")
print(asyncio.run(cg.run("read mol.png")).content)
```

or from a shell, `chemgraph -w ocsr -q "what is in mol.png?"`.

Reading the image is the whole job there. To go further in one conversation, pass the
tools to the general workflow instead:

```python
from chemgraph.graphs.ocsr_agent import DEFAULT_OCSR_TOOLS
from chemgraph.tools.ase_tools import run_ase
from chemgraph.tools.cheminformatics_tools import smiles_to_coordinate_file

ChemGraph(model_name="gpt-4o", workflow_type="single_agent",
          tools=[*DEFAULT_OCSR_TOOLS, smiles_to_coordinate_file, run_ase])
```

## Backends

| `backend=` | what it runs | confidence |
|---|---|---|
| `auto` (default) | DECIMER alone, 0.7 s warm | none |
| `ensemble` | every installed specialist, voted | calibrated from how much they agreed |
| `decimer` / `molnextr` / `molscribe` / `ocsrglyph` | that one | none |
| `alcf` / `shim` | a vision LLM, pick one with `model=` | none |
| `llm` | whichever of those is configured | none |

Only `ensemble` reports a confidence. One model cannot say how likely it is to be
right about a particular image, so nothing is quoted for the others; its overall
benchmark accuracy is available with `report_solo_accuracy=True` for a caller who
knows that is what it means.

To vote a subset of what is installed, name it. The committee is what a calibration
table describes, so a table fit on two models yields a number only when those two
vote:

```python
image_to_smiles_core(img, backend="ensemble",
                     models_wanted=["decimer", "molnextr"],
                     calibration="my_two_model_table.json")
```

The two endpoints use different model-name formats and are not interchangeable: ALCF
takes a HuggingFace path, the Argo shim takes an Argo wire name. Pass the friendly
spelling (`argo:claude-opus-4.8`) and the tool translates.

## Reading the confidence

With `backend="ensemble"` every installed specialist votes, and how much they agree
predicts how often the majority is right. That relationship was measured over 722
benchmark images:

| pattern | meaning | right | measured on |
|---|---|---|---|
| `4` | all four agree | 99.9% | 462 images |
| `3/1` | three agree, one dissents | 98.2% | 135 |
| `2/1/1` | two agree, two differ | 80.2% | 57 |
| `1/1/1/1` | all four differ | 37.7% | 56 |
| `2/2` | two pairs, tied | no number | 12 |

A model that returns something unparseable counts as a dissenting vote, so the parts
of a pattern always add up to the number of models asked.

So a unanimous answer can go straight into a geometry optimization, and a split one
should be checked first. At a 0.95 cut, about 83% of images pass through
automatically. `2/2` has only 12 images behind it, which is under the table's sample
floor, so it reports a label and an interval with no decimal: a 51-point-wide
interval does not support one.

These figures are printed by the table itself, so read them from there instead of
from this page if you have refit it:

```bash
python -c "import json;from importlib import resources;\
print(resources.files('chemgraph.tools').joinpath('ocsr_calibration_4model.json').read_text())"
```

The table ships with ChemGraph and describes this specific set of four models. If you
run a different committee, the tool says so rather than quietly applying numbers that
do not fit.

## Two things worth knowing

**The first call is slow.** Loading a model takes 5-25 s, or 50-66 s for DECIMER;
after that it is 0.3-5 s.
`cold_start` and `latency_s` in the return tell you which happened.

**Check `n_fragments`.** If the image had two molecules, a salt, or a reaction scheme,
the SMILES will contain several disconnected fragments. That still parses and still
looks valid, but building a 3D structure from it places the fragments on top of each
other, and the resulting energy is meaningless while every step reports success. The
tool flags it; do not pass such a result to a calculation without asking which
molecule was meant.

> Specialist models are more accurate than general vision models on ordinary
> structure diagrams, but they fail on unusual drawing styles, Markush structures, and
> reaction schemes. Treat a low confidence as a real signal.
