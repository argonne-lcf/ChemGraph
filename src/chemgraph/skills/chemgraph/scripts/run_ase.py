"""Staged entrypoint for a direct PBS calculation using any supported ASE driver."""

from chemgraph.tools.ase_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
