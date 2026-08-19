"""LangChain tools for generic adsorption and legacy gRASPA calls."""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.adsorption_schema import AdsorptionRequest
from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools.adsorption_core import run_adsorption_core
from chemgraph.tools.graspa_core import (
    _read_graspa_sycl_output,
    mock_graspa,
    run_graspa_core,
)

__all__ = [
    "_read_graspa_sycl_output",
    "mock_graspa",
    "run_adsorption",
    "run_adsorption_core",
    "run_graspa",
    "run_graspa_core",
]


@tool
def run_adsorption(adsorption_input: AdsorptionRequest) -> dict:
    """Run an adsorption simulation with the configured engine."""

    return run_adsorption_core(adsorption_input)


@tool
def run_graspa(graspa_input: graspa_input_schema) -> float:
    """Run one legacy gRASPA simulation and return uptake in mol/kg."""

    result = run_graspa_core(graspa_input)
    if result["status"] != "success":
        raise RuntimeError(result.get("message") or "gRASPA simulation failed")
    return float(result["uptake_in_mol_kg"])
