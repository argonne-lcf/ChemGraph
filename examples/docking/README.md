# Molecular docking with ChemGraph

Runs the AutoDock Vina docking tool through the ChemGraph **`molecular_docking`**
workflow. The agent takes a candidate (SMILES / name / PubChem CID) and a receptor,
and returns the predicted binding affinity and poses.

You can also run it straight from the CLI:
```bash
chemgraph -q "Dock aspirin into 'vancomycin_receptor.pdbqt'" -w molecular_docking
```
or pick **molecular_docking** as the workflow in the Streamlit UI (`streamlit run src/ui/app.py`).

## What's here
- `run_chemgraph.py` — runs the `molecular_docking` workflow to dock a candidate into the receptor.
- `vancomycin_receptor.pdbqt` — a prepared rigid receptor: **vancomycin (chain A of RCSB PDB [1FVM](https://www.rcsb.org/structure/1FVM))**, in PDBQT format (coordinates + Gasteiger charges + AutoDock atom types). PDB data is public domain.

## Setup
```bash
pip install -e ".[docking]"            # ChemGraph + Meeko
conda install -c conda-forge vina      # AutoDock Vina (not pip-installable)
export OPENAI_API_KEY="your_key"       # or another supported provider
```

## Run
```bash
python run_chemgraph.py
```

## Try your own
- Change the **candidate** in `PROMPT` to any SMILES, molecule name, or PubChem CID.
- Point `receptor` at your own prepared `.pdbqt`, or pass a SMILES/name for a small-molecule receptor.
- The search box is chosen automatically (`site_detection="auto"`); override with `center`/`box_size`, or supply a `reference_ligand` to center on a known bound pose.

> Docking scores are approximate — compare trends rather than absolute kcal/mol, and remember that binding is necessary but not sufficient for activity.
