!!! note
    ChemGraph requires **Python 3.11+**.

## Install from PyPI (recommended)

```bash
pip install chemgraph
```

Default installation does not require `tblite`.

To include optional calculator extras (including `tblite`):

```bash
pip install "chemgraph[calculators]"
```

To use the experimental Codex subscription provider with an existing ChatGPT
login, first follow the
[official Codex CLI installation guide](https://learn.chatgpt.com/docs/codex/cli).
The Python SDK does not install the `codex` shell command. Then run:

```bash
pip install "chemgraph[codex]"
codex login
```

See [Experimental Codex subscription support](codex_subscription.md) for usage
and authentication constraints.

!!! warning
    On platforms without a prebuilt `tblite` wheel, installing `calculators` may require a local Fortran toolchain.

## Install from source

### pip/venv

```bash
git clone https://github.com/argonne-lcf/ChemGraph
cd ChemGraph
python -m venv chemgraph-env
source chemgraph-env/bin/activate  # Windows: .\chemgraph-env\Scripts\activate
pip install -e .
```

For experimental Codex subscription support, install the Codex CLI as shown
above, then include the optional extra in the editable install:

```bash
pip install -e ".[codex]"
codex login
```

### conda

```bash
git clone --depth 1 https://github.com/argonne-lcf/ChemGraph
cd ChemGraph
conda env create -f environment.yml
conda activate chemgraph
```

### uv

```bash
git clone https://github.com/argonne-lcf/ChemGraph
cd ChemGraph
uv venv --python 3.11 chemgraph-env
source chemgraph-env/bin/activate  # Windows: .\chemgraph-env\Scripts\activate
uv pip install -e .
```

## Optional UMA install

`uma` and `mace-torch` can conflict through different `e3nn` requirements.
Use separate environments if you need both MACE and UMA.

PyPI attempt:

```bash
pip install "chemgraph[uma]"
```

From source:

```bash
pip install -e ".[uma]"
```

If resolution fails, install UMA in a separate environment dedicated to UMA workflows.
