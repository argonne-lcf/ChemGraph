"""Specialist OCSR worker processes and the scripts that build their environments.

Each worker is a standalone script run by its OWN Python interpreter, not this one.
The four models have mutually incompatible requirements (four Python versions, and a
numpy<2 pin for MolScribe against ChemGraph's numpy 2.x), so each lives in its own
conda environment and is driven over a JSON line protocol on stdin/stdout. See
:mod:`chemgraph.tools.ocsr_worker_client` for the parent side.

Nothing here is imported by ChemGraph. The `*_infer.py` files are package data that
happens to be Python: they are executed as `<their-env-python> <path> --device cpu`,
and importing one into this interpreter would fail on the missing torch or TensorFlow.

The `build_*.sh` scripts create those environments and fetch weights. They are not run
automatically; installing 17.7 GB across four Python versions is a decision the user
makes once, explicitly. See `examples/ocsr/README.md`.
"""
