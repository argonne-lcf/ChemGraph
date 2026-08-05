!!! note
    ChemGraph requires **Python 3.10+**.

## Install from PyPI (recommended)

```bash
pip install chemgraph
```

Default installation does not require `tblite`.

To include optional calculator extras (including `tblite`):

```bash
pip install "chemgraph[calculators]"
```

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

## Optional MACE install

MACE is an optional extra, so a plain install leaves `mace-torch` out and the
`mace_mp`, `mace_off`, and `mace_anicc` calculators stay unregistered:

```bash
pip install "chemgraph[mace]"     # or: pip install -e ".[mace]" from source
```

## Optional UMA install

`uma` and `mace` can conflict through different `e3nn` requirements.
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
