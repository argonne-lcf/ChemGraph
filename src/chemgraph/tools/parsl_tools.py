"""Backward-compatibility alias for :mod:`chemgraph.tools.mace_tools`.

.. deprecated::
    This module has been renamed to ``chemgraph.tools.mace_tools``.
    Import from there instead.  This shim will be removed in a future
    release.
"""

from chemgraph.tools.mace_tools import (  # noqa: F401
    mace_input_schema,
    mace_input_schema_ensemble,
    mace_output_schema,
    run_mace_core,
)
