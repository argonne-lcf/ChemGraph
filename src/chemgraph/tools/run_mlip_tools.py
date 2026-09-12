"""LangChain entry points for calculator-selectable MLIP calculations."""

from langchain_core.tools import tool

from chemgraph.schemas.mlip_input import MLIPBatchInputSchema, MLIPInputSchema
from chemgraph.tools.run_mlip_core import run_mlip_batch_core, run_mlip_core


@tool
def run_mlip(params: MLIPInputSchema) -> dict:
    """Run one MLIP energy calculation or fixed-cell geometry optimization."""
    return run_mlip_core(params)


@tool
def run_mlip_batch(params: MLIPBatchInputSchema) -> dict:
    """Run an ordered batch of MLIP calculations and write a JSON manifest."""
    return run_mlip_batch_core(params)
