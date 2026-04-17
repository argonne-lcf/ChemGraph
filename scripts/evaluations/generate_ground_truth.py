"""Generate a ground-truth evaluation dataset for ChemGraph.

This script builds a JSON file of natural-language chemistry queries
together with their expected tool-call sequences **and actual results**
obtained by executing each tool chain end-to-end.

The tool calls reflect the **current** single-agent tool set:

    molecule_name_to_smiles   -- name -> SMILES
    smiles_to_coordinate_file -- SMILES -> XYZ file on disk
    run_ase                   -- ASE simulation via input_structure_file
    extract_output_json       -- load results from a run_ase output JSON
    calculator                -- safe math expression evaluator (reactions)

Categories of evaluation entries:

    A  Single tool calls (name->SMILES)
    B  Multi-step from molecule name (name->SMILES->coord->run_ase),
       covering all drivers: energy, opt, vib, thermo, dipole
    C  Multi-step from SMILES (SMILES->coord->run_ase), same driver
       coverage as B for parity
    D  Gibbs free energy of reaction calculations (multi-species,
       stoichiometry, name->SMILES->coord->thermo for each species,
       then calculator for the reaction Gibbs free energy expression)

Input file format
-----------------
The ``--input_file`` flag accepts a unified JSON file containing both
molecule data and reaction data::

    {
        "molecules": [
            {"name": "aspirin", "number_of_atoms": 21,
             "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            ...
        ],
        "reactions": [
            {
                "reaction_name": "Methane Combustion",
                "reactants": [
                    {"name": "Methane", "smiles": "C", "coefficient": 1},
                    {"name": "Oxygen",  "smiles": "O=O", "coefficient": 2}
                ],
                "products": [
                    {"name": "Carbon dioxide", "smiles": "O=C=O", "coefficient": 1},
                    {"name": "Water",          "smiles": "O",     "coefficient": 2}
                ]
            },
            ...
        ]
    }

Both ``"molecules"`` and ``"reactions"`` keys are required.
Each reaction species entry **must** include ``"smiles"`` so
the ground truth can encode the expected SMILES lookups.

Scalable query generation via ``--config``
-------------------------------------------
Instead of the hardcoded entry list, a **query-config JSON file** can
be supplied via ``--config`` to control which molecules/reactions map
to which query types.  Each query type specifies a
``molecule_range: [start, end)`` (0-indexed, half-open) into the
``molecules`` array from the input file.  Multi-step queries
additionally specify ``driver``, ``calculator``, and an optional
``temperature``.  Reactions use ``reaction_range`` with cycled
calculators and temperatures.

See ``query_config.json`` for the full schema and a working example.

Query-config schema (abbreviated)::

    {
        "molecule_queries": {
            "molecule_name_to_smiles":    {"molecule_range": [0, 20]},
            "name_to_ase": [
                {"driver": "opt", "calculator": "mace_mp",
                 "molecule_range": [0, 10]},
                {"driver": "thermo", "calculator": "GFN2-xTB",
                 "temperature": 800, "molecule_range": [10, 20]},
                ...
            ],
            "name_to_ase_extract": [...],
            "smiles_to_ase":       [...],
            "smiles_to_ase_extract": [...]
        },
        "reaction_queries": {
            "reaction_range": [0, 10],
            "calculators": ["mace_mp", "GFN2-xTB"],
            "temperatures": [300, 400, 500]
        }
    }

Available calculators: ``"mace_mp"``, ``"GFN2-xTB"`` (alias
``"tblite_gfn2"``).

Available drivers: ``"energy"``, ``"opt"``, ``"vib"``, ``"thermo"``,
``"dipole"``, ``"ir"``.

Usage
-----
    # With a unified input file -- runs tools and captures results
    python generate_ground_truth.py --input_file input_data.json

    # Skip execution (legacy behaviour: empty results)
    python generate_ground_truth.py --input_file input_data.json --skip_execution

    # Custom output path
    python generate_ground_truth.py --input_file input_data.json -o my_gt.json

    # Config-driven scalable generation
    python generate_ground_truth.py --input_file input_data.json --config query_config.json

    # Config-driven, skip execution
    python generate_ground_truth.py --input_file input_data.json --config query_config.json --skip_execution
"""

import argparse
import copy
import json
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---- calculator configs ---------------------------------------------------

MACE_MP = {"calculator_type": "mace_mp", "model": "medium-mpa-0"}
MACE_MP_DESC = "the MACE-MP calculator with the medium-mpa-0 model"
TBLITE_GFN2 = {
    "calculator_type": "TBLite",
    "method": "GFN2-xTB",
}

DRIVER_LABELS = {
    "energy": "single-point energy",
    "vib": "vibrational frequencies",
    "thermo": "thermochemical properties (Gibbs free energy)",
    "dipole": "dipole moment",
    "ir": "infrared spectrum",
}

# Human-readable driver labels used for category derivation.
# These describe the *scientific task*, not internal function names.
DRIVER_CATEGORY_LABELS = {
    "opt": "optimization",
    "vib": "vibrations",
    "thermo": "thermochemistry",
    "dipole": "dipole",
    "energy": "energy",
    "ir": "ir_spectrum",
}


def _derive_category(cfg_key: str, driver: str | None = None) -> str:
    """Derive a human-readable category from the config key and driver.

    Categories describe the *scientific task* and *input type*, not
    internal function names.  For example, a ``name_to_ase`` query with
    ``driver="opt"`` becomes ``"optimization_from_name"``.

    Parameters
    ----------
    cfg_key : str
        The query-config key that generated this entry (e.g.
        ``"molecule_name_to_smiles"``, ``"name_to_ase"``,
        ``"smiles_to_ase_extract"``).
    driver : str or None
        The ASE driver name (e.g. ``"opt"``, ``"vib"``).  Required for
        multi-step simulation entries; ignored for SMILES-lookup and
        reaction entries.

    Returns
    -------
    str
        A category label such as ``"smiles_lookup"``,
        ``"optimization_from_name"``, ``"energy_from_smiles"``, or
        ``"reaction_energy"``.
    """
    if cfg_key == "molecule_name_to_smiles":
        return "smiles_lookup"

    # Determine input type from config key.
    if cfg_key.startswith("name_to_ase"):
        input_type = "from_name"
    elif cfg_key.startswith("smiles_to_ase"):
        input_type = "from_smiles"
    else:
        return cfg_key  # fallback

    driver_label = DRIVER_CATEGORY_LABELS.get(driver, driver or "unknown")
    return f"{driver_label}_{input_type}"


# ---- tool-call dict helpers ------------------------------------------------


def _run_ase_tool_call(
    input_structure_file: str,
    driver: str,
    calculator: dict,
    temperature: float | None = None,
) -> dict:
    """Build a ground-truth ``run_ase`` tool call dict.

    Only scientifically-relevant parameters are included; schema
    defaults (optimizer, fmax, steps, pressure, output_results_file)
    are left for the evaluator to fill via ``apply_defaults``.
    """
    params: dict = {
        "input_structure_file": input_structure_file,
        "driver": driver,
        "calculator": calculator,
    }
    if temperature is not None:
        params["temperature"] = temperature
    return {"run_ase": {"params": params}}


# ---- query builders --------------------------------------------------------
# Each builder returns {"query": str, "tool_calls": list[dict]}.


def _build_simulation_query(
    molecule_label: str,
    driver: str,
    temperature: float | None,
    calc_description: str,
    *,
    label_is_smiles: bool = False,
) -> str:
    """Build a natural-language query string for an ASE simulation.

    Parameters
    ----------
    molecule_label : str
        The molecule name or SMILES string used in the query.
    driver : str
        ASE driver name (``"energy"``, ``"opt"``, ``"vib"``, etc.).
    temperature : float or None
        Temperature in Kelvin (included when not ``None``).
    calc_description : str
        Human-readable calculator label for the query.
    label_is_smiles : bool
        If ``True``, the query says "the molecule with SMILES: ...".
    """
    temp_str = f" at {int(temperature)} K" if temperature else ""
    mol_ref = (
        f"the molecule with SMILES: {molecule_label}"
        if label_is_smiles
        else molecule_label
    )

    if driver == "opt":
        if label_is_smiles:
            return (
                f"Run geometry optimization{temp_str} using {calc_description} "
                f"for {mol_ref} and report its energy."
            )
        return (
            f"Run geometry optimization for {mol_ref}{temp_str} "
            f"and report its energy using {calc_description}."
        )

    driver_label = DRIVER_LABELS.get(driver, driver)
    if label_is_smiles:
        return (
            f"Report the {driver_label}{temp_str} using {calc_description} "
            f"for {mol_ref}."
        )
    return f"Report the {driver_label} of {mol_ref}{temp_str} using {calc_description}."


def build_name_to_smiles(molecules: list[dict], count: int = 1) -> dict:
    """Name -> SMILES for *count* molecules."""
    selected = molecules[:count]
    if count == 1:
        query = (
            f"Provide the SMILES string corresponding to this molecule: "
            f"{selected[0]['name']}"
        )
    else:
        names = " and ".join(m["name"] for m in selected)
        query = f"Provide the SMILES string corresponding to these molecules: {names}"
    tool_calls = [{"molecule_name_to_smiles": {"name": m["name"]}} for m in selected]
    return {"query": query, "tool_calls": tool_calls}


def build_name_to_ase(
    molecule: dict,
    driver: str,
    calculator: dict,
    temperature: float | None = None,
    calc_description: str = "",
    *,
    extract: bool = False,
) -> dict:
    """Multi-step: name -> SMILES -> coordinate file -> run_ase [-> extract]."""
    query = _build_simulation_query(
        molecule["name"], driver, temperature, calc_description
    )
    tool_calls = [
        {"molecule_name_to_smiles": {"name": molecule["name"]}},
        {"smiles_to_coordinate_file": {"smiles": molecule["smiles"]}},
        _run_ase_tool_call("molecule.xyz", driver, calculator, temperature),
    ]
    if extract:
        tool_calls.append({"extract_output_json": {"json_file": "output.json"}})
    return {"query": query, "tool_calls": tool_calls}


def build_smiles_to_ase(
    molecule: dict,
    driver: str,
    calculator: dict,
    temperature: float | None = None,
    calc_description: str = "",
    *,
    extract: bool = False,
) -> dict:
    """Multi-step: SMILES -> coordinate file -> run_ase [-> extract]."""
    query = _build_simulation_query(
        molecule["smiles"],
        driver,
        temperature,
        calc_description,
        label_is_smiles=True,
    )
    tool_calls = [
        {"smiles_to_coordinate_file": {"smiles": molecule["smiles"]}},
        _run_ase_tool_call("molecule.xyz", driver, calculator, temperature),
    ]
    if extract:
        tool_calls.append({"extract_output_json": {"json_file": "output.json"}})
    return {"query": query, "tool_calls": tool_calls}


def build_reaction_gibbs_free_energy(
    reaction: dict,
    calculator: dict,
    temperature: float,
    calc_description: str = "",
) -> dict:
    """Build a Gibbs-free-energy-of-reaction evaluation entry.

    The expected tool-call sequence is, for each unique species:

    1. ``molecule_name_to_smiles``
    2. ``smiles_to_coordinate_file``
    3. ``run_ase`` (driver="thermo", with temperature)

    followed by a final:

    4. ``calculator`` with the deltaG expression
       ``deltaG = sum coeff_i * G_product_i - sum coeff_j * G_reactant_j``

    The per-species steps are interleaved so that each coordinate file
    is consumed by ``run_ase`` immediately after it is written, avoiding
    the file-overwrite problem that would occur if all writes were
    batched before all thermochemistry calculations.

    Parameters
    ----------
    reaction : dict
        A reaction dict with keys ``"reaction_name"``, ``"reactants"``
        and ``"products"``.  Each species entry has ``"name"``,
        ``"smiles"``, and ``"coefficient"``.
    calculator : dict
        Calculator config dict (e.g. ``MACE_MP``).
    temperature : float
        Temperature in Kelvin for thermochemistry calculations.
    calc_description : str
        Human-readable calculator label for the query string.

    Returns
    -------
    dict
        ``{"query": str, "tool_calls": list[dict]}``
    """
    rxn_name = reaction["reaction_name"]
    reactants = reaction["reactants"]
    products = reaction["products"]

    # Collect unique species in order (reactants first, then products).
    seen: set[str] = set()
    unique_species: list[dict] = []
    for species in reactants + products:
        if species["name"] not in seen:
            seen.add(species["name"])
            unique_species.append(species)

    # Build query string.
    query = (
        f"Report the Gibbs free energy of reaction for {rxn_name} "
        f"at {int(temperature)} K using {calc_description}. "
        f"Report the energy in eV. "
        f"The balanced reaction is: "
    )
    reactant_strs = [
        f"{s['coefficient']} {s['name']}" if s["coefficient"] != 1 else s["name"]
        for s in reactants
    ]
    product_strs = [
        f"{s['coefficient']} {s['name']}" if s["coefficient"] != 1 else s["name"]
        for s in products
    ]
    query += " + ".join(reactant_strs) + " -> " + " + ".join(product_strs)

    # Build tool calls — interleaved per species so each coordinate
    # file is immediately consumed before the next species overwrites it.
    tool_calls: list[dict] = []

    for species in unique_species:
        tool_calls.append({"molecule_name_to_smiles": {"name": species["name"]}})
        tool_calls.append({"smiles_to_coordinate_file": {"smiles": species["smiles"]}})
        tool_calls.append(
            _run_ase_tool_call(
                input_structure_file="molecule.xyz",
                driver="thermo",
                calculator=calculator,
                temperature=temperature,
            )
        )

    # Final step: calculator expression for deltaG
    product_terms = [
        f"{s['coefficient']}*G_{s['name'].replace(' ', '_')}" for s in products
    ]
    reactant_terms = [
        f"{s['coefficient']}*G_{s['name'].replace(' ', '_')}" for s in reactants
    ]
    expression = (
        "("
        + " + ".join(product_terms)
        + ")"
        + " - "
        + "("
        + " + ".join(reactant_terms)
        + ")"
    )
    tool_calls.append({"calculator": {"expression": expression}})

    return {"query": query, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Tool execution engine
# ---------------------------------------------------------------------------


def _import_tools() -> dict:
    """Lazily import ChemGraph tools (heavy dependencies).

    Returns
    -------
    dict
        Mapping of tool function name -> LangChain tool object.
    """
    from chemgraph.tools.cheminformatics_tools import (
        molecule_name_to_smiles,
        smiles_to_coordinate_file,
    )
    from chemgraph.tools.ase_tools import run_ase, extract_output_json
    from chemgraph.tools.generic_tools import calculator

    return {
        "molecule_name_to_smiles": molecule_name_to_smiles,
        "smiles_to_coordinate_file": smiles_to_coordinate_file,
        "run_ase": run_ase,
        "extract_output_json": extract_output_json,
        "calculator": calculator,
    }


def _execute_tool_call(
    tool_name: str,
    tool_args: dict,
    tools: dict,
) -> dict | str:
    """Invoke a single LangChain tool and return the raw result.

    Parameters
    ----------
    tool_name : str
        One of the tool function names.
    tool_args : dict
        Arguments to pass to the tool via ``.invoke()``.
    tools : dict
        Mapping of tool name -> LangChain tool object.

    Returns
    -------
    dict | str
        The tool's return value, or an error dict on failure.
    """
    tool_fn = tools.get(tool_name)
    if tool_fn is None:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    try:
        return tool_fn.invoke(tool_args)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _execute_entry(
    entry: dict,
    tools: dict,
    work_dir: str,
) -> list[dict]:
    """Execute all tool calls for a single evaluation entry sequentially.

    Each tool call is executed in *work_dir* so that intermediate files
    (``molecule.xyz``, ``output.json``, etc.) are written there and do
    not clash between entries.

    For reaction-energy entries (Category D) the symbolic calculator
    expression (e.g. ``(1*E_Water) - (1*E_Methane)``) is resolved by
    substituting actual energies obtained from the preceding ``run_ase``
    calls before invoking the ``calculator`` tool.

    Parameters
    ----------
    entry : dict
        An evaluation entry with ``"tool_calls"`` list.
    tools : dict
        Tool name -> LangChain tool object mapping.
    work_dir : str
        Temporary working directory for this entry.

    Returns
    -------
    list[dict]
        One result dict per tool call, in the same order:
        ``{"tool": str, "input": dict, "output": <tool_result>}``
    """
    original_cwd = os.getcwd()
    os.chdir(work_dir)

    # Set CHEMGRAPH_LOG_DIR so _resolve_path writes files into work_dir.
    old_log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    os.environ["CHEMGRAPH_LOG_DIR"] = work_dir

    # Pre-scan: detect reaction entries so we can track species
    # Gibbs free energies for the final calculator substitution.
    #
    # Reaction entries follow an interleaved pattern:
    #   (molecule_name_to_smiles, smiles_to_coordinate_file, run_ase) * N
    #   calculator (symbolic expression)
    #
    # Each molecule_name_to_smiles immediately precedes its species'
    # coordinate generation and thermo calculation, so we track the
    # most recently seen species name.
    species_energies: dict[str, float] = {}
    is_reaction_entry = _is_reaction_entry(entry["tool_calls"])

    current_species_name: str | None = None
    results: list[dict] = []
    try:
        for tc in entry["tool_calls"]:
            tool_name, tool_args = next(iter(tc.items()))

            # Track current species name from molecule_name_to_smiles.
            if is_reaction_entry and tool_name == "molecule_name_to_smiles":
                current_species_name = tool_args.get("name")

            # For reaction entries: substitute real energies into the
            # symbolic calculator expression before executing it.
            if is_reaction_entry and tool_name == "calculator" and species_energies:
                tool_args = _substitute_energies(tool_args, species_energies)

            result = _execute_tool_call(tool_name, tool_args, tools)

            # Track Gibbs free energies from run_ase thermo results
            # for reaction entries.
            if is_reaction_entry and tool_name == "run_ase":
                if (
                    current_species_name is not None
                    and isinstance(result, dict)
                    and result.get("status") == "success"
                ):
                    key = f"G_{current_species_name.replace(' ', '_')}"
                    thermo = result.get("result", {}).get("thermochemistry", {})
                    species_energies[key] = thermo["gibbs_free_energy"]

            results.append({"tool": tool_name, "input": tool_args, "output": result})
    finally:
        os.chdir(original_cwd)
        if old_log_dir is None:
            os.environ.pop("CHEMGRAPH_LOG_DIR", None)
        else:
            os.environ["CHEMGRAPH_LOG_DIR"] = old_log_dir

    return results


def _is_reaction_entry(tool_calls: list[dict]) -> bool:
    """Return True if *tool_calls* matches the reaction calculation pattern.

    The interleaved pattern is::

        (molecule_name_to_smiles, smiles_to_coordinate_file, run_ase) * N
        + calculator

    where N >= 1 is the number of unique species.
    """
    if not tool_calls:
        return False
    names = [next(iter(tc)) for tc in tool_calls]
    if names[-1] != "calculator":
        return False
    # The body (everything except the trailing calculator) must be
    # a repetition of the 3-tool triplet.
    body = names[:-1]
    if len(body) == 0 or len(body) % 3 != 0:
        return False
    triplet = ["molecule_name_to_smiles", "smiles_to_coordinate_file", "run_ase"]
    for i in range(0, len(body), 3):
        if body[i : i + 3] != triplet:
            return False
    return True


def _substitute_energies(
    tool_args: dict,
    energies: dict[str, float],
) -> dict:
    """Replace symbolic energy variables in a calculator expression.

    Parameters
    ----------
    tool_args : dict
        Original calculator args, e.g.
        ``{"expression": "(1*G_Water) - (1*G_Methane)"}``.
    energies : dict[str, float]
        Mapping of variable names to numeric values, e.g.
        ``{"G_Water": -14.23, "G_Methane": -24.05}``.

    Returns
    -------
    dict
        New args dict with variables replaced by their numeric values.
    """
    expr = tool_args.get("expression", "")
    for var, val in energies.items():
        # Use parenthesised value to handle negative numbers correctly.
        expr = expr.replace(var, f"({val})")
    return {**tool_args, "expression": expr}


def _result_to_structured_output(entry: dict, final_result) -> dict | None:
    """Convert a raw tool-call result to a ``ResponseFormatter``-compatible dict.

    Maps the tool-specific output format to the schema used by the
    agent's structured output (``ResponseFormatter`` from
    ``chemgraph.schemas.agent_response``).

    Parameters
    ----------
    entry : dict
        The evaluation entry containing ``"tool_calls"`` and ``"query"``.
    final_result
        The final result value produced by tool execution.

    Returns
    -------
    dict or None
        A dict matching the ``ResponseFormatter`` schema with keys
        ``smiles``, ``scalar_answer``, ``dipole``,
        ``vibrational_answer``, ``ir_spectrum``, and ``atoms_data``.
        Returns ``None`` if the result cannot be mapped (e.g. error
        results).
    """
    if final_result is None or final_result == "":
        return None

    # Error results cannot be mapped.
    if isinstance(final_result, dict) and final_result.get("status") == "error":
        return None

    structured: dict = {
        "smiles": None,
        "scalar_answer": None,
        "dipole": None,
        "vibrational_answer": None,
        "ir_spectrum": None,
        "atoms_data": None,
    }

    tool_calls = entry["tool_calls"]
    last_tool = next(iter(tool_calls[-1]))

    if last_tool == "molecule_name_to_smiles":
        # SMILES lookup result(s).
        if isinstance(final_result, list):
            structured["smiles"] = [r.get("smiles", str(r)) for r in final_result]
        elif isinstance(final_result, dict):
            structured["smiles"] = [final_result.get("smiles", str(final_result))]
        elif isinstance(final_result, str):
            structured["smiles"] = [final_result]

    elif last_tool == "run_ase":
        driver = _get_last_driver(tool_calls)
        if not isinstance(final_result, dict):
            return None  # cannot map non-dict result
        elif driver in ("energy", "opt"):
            energy = final_result.get("single_point_energy")
            if energy is not None:
                prop = (
                    "single-point energy" if driver == "energy" else "optimized energy"
                )
                structured["scalar_answer"] = {
                    "value": energy,
                    "property": prop,
                    "unit": final_result.get("unit", "eV"),
                }
        elif driver == "vib":
            nested = final_result.get("result", {})
            vib_data = nested.get("vibrational_frequencies", {})
            freqs = vib_data.get("frequencies", [])
            if freqs:
                structured["vibrational_answer"] = {
                    "frequency_cm1": [str(f) for f in freqs],
                }
        elif driver == "thermo":
            nested = final_result.get("result", {})
            thermo = nested.get("thermochemistry", {})
            gfe = thermo.get("gibbs_free_energy")
            if gfe is not None:
                structured["scalar_answer"] = {
                    "value": gfe,
                    "property": "Gibbs free energy",
                    "unit": thermo.get("unit", "eV"),
                }
        elif driver == "dipole":
            dipole_moment = final_result.get("dipole_moment")
            if dipole_moment is not None:
                structured["dipole"] = {
                    "value": dipole_moment,
                    "unit": "e * Angstrom",
                }
        elif driver == "ir":
            nested = final_result.get("result", {})
            ir_data = nested.get("ir_data", nested.get("ir", {}))
            freqs = ir_data.get("frequencies", [])
            intensities = ir_data.get("intensities", [])
            if freqs:
                structured["ir_spectrum"] = {
                    "frequency_cm1": [str(f) for f in freqs],
                    "intensity": [str(i) for i in intensities],
                }
        else:
            return None  # Unknown driver

    elif last_tool == "calculator":
        # Reaction energy (deltaG).
        try:
            value = (
                float(final_result) if isinstance(final_result, str) else final_result
            )
        except (ValueError, TypeError):
            return None
        else:
            structured["scalar_answer"] = {
                "value": value,
                "property": "Gibbs free energy of reaction",
                "unit": "eV",
            }

    elif last_tool == "extract_output_json":
        # Full output JSON — the driver determines which field to
        # extract.  This mirrors the ``run_ase`` handler above because
        # ``extract_output_json`` returns the same result structure
        # (just read from file instead of returned inline).
        if not isinstance(final_result, dict):
            return None
        driver = _get_last_driver(tool_calls)
        if driver in ("energy", "opt"):
            energy = final_result.get("single_point_energy")
            if energy is not None:
                prop = (
                    "single-point energy" if driver == "energy" else "optimized energy"
                )
                structured["scalar_answer"] = {
                    "value": energy,
                    "property": prop,
                    "unit": final_result.get("energy_unit", "eV"),
                }
        elif driver == "thermo":
            nested = final_result.get("result", {})
            thermo = nested.get("thermochemistry", {})
            gfe = thermo.get("gibbs_free_energy")
            if gfe is not None:
                structured["scalar_answer"] = {
                    "value": gfe,
                    "property": "Gibbs free energy",
                    "unit": thermo.get("unit", "eV"),
                }
        elif driver == "vib":
            nested = final_result.get("result", {})
            vib_data = nested.get("vibrational_frequencies", {})
            freqs = vib_data.get("frequencies", [])
            if freqs:
                structured["vibrational_answer"] = {
                    "frequency_cm1": [str(f) for f in freqs],
                }
        elif driver == "dipole":
            dipole_moment = final_result.get("dipole_moment")
            if dipole_moment is not None:
                structured["dipole"] = {
                    "value": dipole_moment,
                    "unit": "e * Angstrom",
                }
        elif driver == "ir":
            nested = final_result.get("result", {})
            ir_data = nested.get("ir_data", nested.get("ir", {}))
            freqs = ir_data.get("frequencies", [])
            intensities = ir_data.get("intensities", [])
            if freqs:
                structured["ir_spectrum"] = {
                    "frequency_cm1": [str(f) for f in freqs],
                    "intensity": [str(i) for i in intensities],
                }

    else:
        return None  # Unknown tool

    return structured


def _get_last_driver(tool_calls: list[dict]) -> str | None:
    """Extract the ``driver`` argument from the last ``run_ase`` call.

    Scans *tool_calls* in reverse to find the most recent ``run_ase``
    entry and returns its ``driver`` value.

    Returns
    -------
    str or None
        The driver string (e.g. ``"energy"``, ``"opt"``, ``"vib"``),
        or ``None`` if no ``run_ase`` call is found.
    """
    for tc in reversed(tool_calls):
        if "run_ase" in tc:
            params = tc["run_ase"].get("params", tc["run_ase"])
            return params.get("driver")
    return None


def _make_serialisable(obj):
    """Recursively convert an object to JSON-serialisable types.

    Handles Pydantic models, numpy scalars/arrays, NaN floats, and
    other non-standard types that ``json.dump`` would reject.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return _make_serialisable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v  # NaN -> None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if obj != obj else obj  # NaN -> None
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    # Pydantic models
    if hasattr(obj, "model_dump"):
        return _make_serialisable(obj.model_dump())
    if hasattr(obj, "dict"):
        return _make_serialisable(obj.dict())
    return str(obj)


# ---------------------------------------------------------------------------
# Entry generation
# ---------------------------------------------------------------------------


def _build_entries(
    molecules: list[dict],
    reactions: list[dict],
) -> list[dict]:
    """Build the list of evaluation entries (query + tool_calls only).

    Parameters
    ----------
    molecules : list[dict]
        Molecule dicts.  At least 6 are required.
    reactions : list[dict]
        Reaction dicts.

    Returns
    -------
    list[dict]
        Raw entries with ``"query"`` and ``"tool_calls"`` keys.
    """
    if len(molecules) < 6:
        raise ValueError(
            f"Need at least 6 molecules, got {len(molecules)}. "
            "Provide a larger input dataset."
        )

    entries: list[dict] = []

    def _tag(entry: dict, category: str) -> dict:
        entry["category"] = category
        return entry

    # ---- Category A: single tool calls ------------------------------------
    # 1. Name -> SMILES (1 molecule)
    entries.append(_tag(build_name_to_smiles([molecules[0]], count=1), "smiles_lookup"))

    # 2. Name -> SMILES (2 molecules)
    entries.append(_tag(build_name_to_smiles(molecules[0:2], count=2), "smiles_lookup"))

    # ---- Category B: multi-step from molecule name -----------------------
    # 3. Name -> coord -> opt (MACE)
    entries.append(
        _tag(
            build_name_to_ase(
                molecules[0], "opt", MACE_MP, calc_description=MACE_MP_DESC
            ),
            "optimization_from_name",
        )
    )

    # 4. Name -> coord -> vib (MACE)
    entries.append(
        _tag(
            build_name_to_ase(
                molecules[2], "vib", MACE_MP, calc_description=MACE_MP_DESC
            ),
            "vibrations_from_name",
        )
    )

    # 5. Name -> coord -> thermo (TBLite GFN2-xTB, 800 K)
    entries.append(
        _tag(
            build_name_to_ase(
                molecules[3],
                "thermo",
                TBLITE_GFN2,
                temperature=800,
                calc_description="GFN2-xTB",
            ),
            "thermochemistry_from_name",
        )
    )

    # 6. Name -> coord -> dipole (TBLite GFN2-xTB)
    entries.append(
        _tag(
            build_name_to_ase(
                molecules[4], "dipole", TBLITE_GFN2, calc_description="GFN2-xTB"
            ),
            "dipole_from_name",
        )
    )

    # 7. Name -> coord -> energy (MACE)
    entries.append(
        _tag(
            build_name_to_ase(
                molecules[5], "energy", MACE_MP, calc_description=MACE_MP_DESC
            ),
            "energy_from_name",
        )
    )

    # 8. Name -> coord -> energy -> extract results (MACE)
    entries.append(
        _tag(
            build_name_to_ase(
                molecules[5],
                "energy",
                MACE_MP,
                calc_description=MACE_MP_DESC,
                extract=True,
            ),
            "energy_from_name",
        )
    )

    # ---- Category C: multi-step from SMILES ------------------------------
    # 9. SMILES -> coord -> opt (MACE)
    entries.append(
        _tag(
            build_smiles_to_ase(
                molecules[0], "opt", MACE_MP, calc_description=MACE_MP_DESC
            ),
            "optimization_from_smiles",
        )
    )

    # 10. SMILES -> coord -> vib (MACE)
    entries.append(
        _tag(
            build_smiles_to_ase(
                molecules[2], "vib", MACE_MP, calc_description=MACE_MP_DESC
            ),
            "vibrations_from_smiles",
        )
    )

    # 11. SMILES -> coord -> thermo (TBLite GFN2-xTB, 800 K)
    entries.append(
        _tag(
            build_smiles_to_ase(
                molecules[3],
                "thermo",
                TBLITE_GFN2,
                temperature=800,
                calc_description="GFN2-xTB",
            ),
            "thermochemistry_from_smiles",
        )
    )

    # 12. SMILES -> coord -> dipole (TBLite GFN2-xTB)
    entries.append(
        _tag(
            build_smiles_to_ase(
                molecules[4], "dipole", TBLITE_GFN2, calc_description="GFN2-xTB"
            ),
            "dipole_from_smiles",
        )
    )

    # 13. SMILES -> coord -> energy (MACE)
    entries.append(
        _tag(
            build_smiles_to_ase(
                molecules[5], "energy", MACE_MP, calc_description=MACE_MP_DESC
            ),
            "energy_from_smiles",
        )
    )

    # 14. SMILES -> coord -> opt -> extract results (TBLite GFN2-xTB)
    entries.append(
        _tag(
            build_smiles_to_ase(
                molecules[4],
                "opt",
                TBLITE_GFN2,
                calc_description="GFN2-xTB",
                extract=True,
            ),
            "optimization_from_smiles",
        )
    )

    # ---- Category D: Gibbs free energy of reaction calculations ------------
    reaction_calcs = [
        (MACE_MP, MACE_MP_DESC),
        (TBLITE_GFN2, "GFN2-xTB"),
    ]
    reaction_temperatures = [300.0, 400.0, 500.0]
    for rxn_idx, rxn in enumerate(reactions):
        calc, calc_desc = reaction_calcs[rxn_idx % len(reaction_calcs)]
        temp = reaction_temperatures[rxn_idx % len(reaction_temperatures)]
        entries.append(
            _tag(
                build_reaction_gibbs_free_energy(
                    rxn, calc, temperature=temp, calc_description=calc_desc
                ),
                "reaction_energy",
            )
        )

    return entries


# ---------------------------------------------------------------------------
# Config-driven entry generation
# ---------------------------------------------------------------------------

CALCULATOR_REGISTRY: dict[str, dict] = {
    "mace_mp": MACE_MP,
    "GFN2-xTB": TBLITE_GFN2,
    "tblite_gfn2": TBLITE_GFN2,
}


def _resolve_calculator(name: str) -> tuple[dict, str]:
    """Map a human-readable calculator name to its config dict.

    Parameters
    ----------
    name : str
        Calculator identifier, e.g. ``"mace_mp"`` or ``"GFN2-xTB"``.

    Returns
    -------
    tuple[dict, str]
        ``(calculator_config, description)`` — the config dict suitable
        for ``_run_ase_tool_call`` and a human-readable label for query
        strings.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    cfg = CALCULATOR_REGISTRY.get(name)
    if cfg is None:
        raise ValueError(
            f"Unknown calculator {name!r}. Available: {list(CALCULATOR_REGISTRY)}"
        )
    # Build a human-readable description, including the model name
    # when one is configured.
    desc = name
    model = cfg.get("model")
    if model:
        desc = f"the {name} calculator with the {model} model"
    return cfg, desc


def _resolve_molecule_range(
    molecules: list[dict],
    molecule_range: list[int],
    label: str = "",
) -> list[dict]:
    """Slice *molecules* by a ``[start, end)`` range, clamping to bounds.

    Parameters
    ----------
    molecules : list[dict]
        Full list of molecule dicts from the input file.
    molecule_range : list[int]
        Two-element list ``[start, end)`` (half-open, 0-indexed).
    label : str
        Human-readable label for warning messages.

    Returns
    -------
    list[dict]
        The selected molecules.  May be empty if the range is entirely
        out of bounds.
    """
    start, end = molecule_range
    n = len(molecules)
    if start >= n:
        log.warning(
            "%s: molecule_range [%d, %d) is entirely out of bounds "
            "(only %d molecules available). Skipping.",
            label,
            start,
            end,
            n,
        )
        return []
    if end > n:
        log.warning(
            "%s: molecule_range [%d, %d) exceeds available molecules "
            "(%d). Clamping to [%d, %d).",
            label,
            start,
            end,
            n,
            start,
            n,
        )
        end = n
    return molecules[start:end]


def _build_entries_from_config(
    molecules: list[dict],
    reactions: list[dict],
    config: dict,
) -> list[dict]:
    """Build evaluation entries driven by a query-config dict.

    The config dict has two top-level keys:

    ``molecule_queries``
        Controls which molecules are used for each query type.
    ``reaction_queries``
        Controls which reactions to include and how to cycle
        calculators / temperatures.

    See ``query_config.json`` for the full schema and examples.

    Parameters
    ----------
    molecules : list[dict]
        Molecule dicts from the input file.
    reactions : list[dict]
        Reaction dicts from the input file.
    config : dict
        The parsed query-config JSON.

    Returns
    -------
    list[dict]
        Raw entries with ``"query"`` and ``"tool_calls"`` keys.
    """
    entries: list[dict] = []
    mol_cfg = config.get("molecule_queries", {})
    rxn_cfg = config.get("reaction_queries", {})

    # ---- Category A: single tool calls -----------------------------------

    # molecule_name_to_smiles
    nts_cfg = mol_cfg.get("molecule_name_to_smiles")
    if nts_cfg is not None:
        mol_range = nts_cfg.get("molecule_range", [0, 0])
        selected = _resolve_molecule_range(
            molecules, mol_range, "molecule_name_to_smiles"
        )
        for mol in selected:
            entry = build_name_to_smiles([mol], count=1)
            entry["category"] = _derive_category("molecule_name_to_smiles")
            entries.append(entry)

    # ---- Categories B & C: multi-step simulation queries -------------------
    #
    # Each config key maps to a builder function and an optional extract
    # flag.  The loop body is identical for all four variants.
    _multistep_specs: list[tuple[str, callable, bool]] = [
        ("name_to_ase", build_name_to_ase, False),
        ("name_to_ase_extract", build_name_to_ase, True),
        ("smiles_to_ase", build_smiles_to_ase, False),
        ("smiles_to_ase_extract", build_smiles_to_ase, True),
    ]

    for cfg_key, builder_fn, do_extract in _multistep_specs:
        for spec in mol_cfg.get(cfg_key, []):
            driver = spec["driver"]
            calc_cfg, calc_desc = _resolve_calculator(spec["calculator"])
            temperature = spec.get("temperature")
            mol_range = spec.get("molecule_range", [0, 0])
            selected = _resolve_molecule_range(
                molecules,
                mol_range,
                f"{cfg_key}({driver}/{spec['calculator']})",
            )
            category = _derive_category(cfg_key, driver)
            for mol in selected:
                entry = builder_fn(
                    mol,
                    driver,
                    calc_cfg,
                    temperature=temperature,
                    calc_description=calc_desc,
                    extract=do_extract,
                )
                entry["category"] = category
                entries.append(entry)

    # ---- Category D: Gibbs free energy of reaction -----------------------

    if rxn_cfg:
        rxn_range = rxn_cfg.get("reaction_range", [0, len(reactions)])
        calc_names = rxn_cfg.get("calculators", ["mace_mp", "GFN2-xTB"])
        temps = rxn_cfg.get("temperatures", [300.0, 400.0, 500.0])

        start, end = rxn_range
        n_rxn = len(reactions)
        if start >= n_rxn:
            log.warning(
                "reaction_range [%d, %d) is entirely out of bounds "
                "(only %d reactions available). Skipping reactions.",
                start,
                end,
                n_rxn,
            )
        else:
            if end > n_rxn:
                log.warning(
                    "reaction_range [%d, %d) exceeds available reactions "
                    "(%d). Clamping to [%d, %d).",
                    start,
                    end,
                    n_rxn,
                    start,
                    n_rxn,
                )
                end = n_rxn

            # Resolve calculator configs once.
            resolved_calcs = [_resolve_calculator(c) for c in calc_names]

            selected_rxns = reactions[start:end]
            for rxn_idx, rxn in enumerate(selected_rxns):
                calc_cfg, calc_desc = resolved_calcs[rxn_idx % len(resolved_calcs)]
                temp = temps[rxn_idx % len(temps)]
                entry = build_reaction_gibbs_free_energy(
                    rxn,
                    calc_cfg,
                    temperature=temp,
                    calc_description=calc_desc,
                )
                entry["category"] = "reaction_energy"
                entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_ground_truth(
    molecules: list[dict],
    reactions: list[dict],
    *,
    execute: bool = True,
    config: dict | None = None,
) -> list[dict]:
    """Build the full evaluation dataset, optionally running tools.

    Parameters
    ----------
    molecules : list[dict]
        List of molecule dicts.  At least 6 are required when no
        *config* is supplied (legacy mode).
    reactions : list[dict]
        Reaction dicts.
    execute : bool
        If ``True`` (default), each tool-call chain is executed and the
        results are captured in ``answer.result``.  If ``False``,
        ``answer.result`` is set to ``""`` (legacy behaviour).
    config : dict or None
        If provided, entries are built from the query-config mapping
        instead of the hardcoded ``_build_entries`` logic.  This
        enables scalable evaluation — see ``query_config.json`` for
        the expected schema.

    Returns
    -------
    list[dict]
        Evaluation entries with ``id``, ``query``, and ``answer`` keys.
    """
    if config is not None:
        entries = _build_entries_from_config(molecules, reactions, config)
    else:
        entries = _build_entries(molecules, reactions=reactions)

    tools = None
    base_tmp_dir = None

    if execute:
        log.info("Importing ChemGraph tools ...")
        tools = _import_tools()
        base_tmp_dir = tempfile.mkdtemp(prefix="chemgraph_gt_")
        log.info("Temp directory for execution: %s", base_tmp_dir)

    dataset: list[dict] = []

    for idx, entry in enumerate(entries, start=1):
        entry_id = str(idx)
        query_preview = entry["query"]
        log.info("[%d/%d] %s", idx, len(entries), query_preview)

        # Deep-copy tool_calls *before* execution -- tool invocation may
        # mutate dicts in-place (e.g. Pydantic validation replacing a
        # calculator dict with a MaceCalc object).
        tool_calls_snapshot = copy.deepcopy(entry["tool_calls"])

        result_data: list[dict] | str = ""

        if execute and tools is not None and base_tmp_dir is not None:
            # Each entry gets its own temp directory so files don't collide.
            entry_dir = os.path.join(base_tmp_dir, f"entry_{entry_id}")
            os.makedirs(entry_dir, exist_ok=True)

            try:
                step_results = _execute_entry(entry, tools, entry_dir)
                result_data = _make_serialisable(step_results)

                # Patch: for reaction-energy entries, update the
                # symbolic calculator expression in tool_calls_snapshot
                # with the actual numeric expression used during
                # execution so the final JSON contains real values.
                if _is_reaction_entry(entry["tool_calls"]):
                    for step in result_data:
                        if step.get("tool") == "calculator":
                            numeric_expr = step["input"]["expression"]
                            for tc in tool_calls_snapshot:
                                if "calculator" in tc:
                                    tc["calculator"]["expression"] = numeric_expr
                                    break
                            break

                log.info("  -> OK (%d steps executed)", len(step_results))
            except Exception as exc:
                log.warning("  -> FAILED: %s", exc)
                result_data = _make_serialisable(
                    {
                        "status": "error",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        # Extract final result.
        if isinstance(result_data, list) and len(result_data) > 0:
            # Check if all tool calls use the same tool (parallel
            # independent calls, e.g. two molecule_name_to_smiles).
            tool_names = {step["tool"] for step in result_data}
            if len(tool_names) == 1 and len(result_data) > 1:
                # All calls are independent invocations of the same
                # tool — include every output so the answer reflects
                # all molecules in the query.
                final_result = [step.get("output", step) for step in result_data]
            else:
                # Multi-step pipeline — the last step's output is the
                # final answer.
                final_result = result_data[-1].get("output", result_data[-1])
        else:
            final_result = result_data

        # Build structured output (ResponseFormatter-compatible dict).
        structured_output = _result_to_structured_output(entry, final_result)

        answer_dict: dict = {
            "tool_calls": tool_calls_snapshot,
            "result": final_result,
        }
        if structured_output is not None:
            answer_dict["structured_output"] = structured_output

        dataset.append(
            {
                "id": entry_id,
                "category": entry.get("category", ""),
                "query": entry["query"],
                "answer": answer_dict,
            }
        )

    if base_tmp_dir is not None:
        log.info("Cleaning up temp directory: %s", base_tmp_dir)
        shutil.rmtree(base_tmp_dir, ignore_errors=True)

    return dataset


# ---- CLI -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate a ground-truth evaluation dataset for ChemGraph."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help=(
            "Path to a unified JSON file with molecule and reaction data. "
            'Expected format: {"molecules": [...], "reactions": [...]}.'
        ),
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="ground_truth.json",
        help="Path to the output ground-truth JSON file.",
    )
    parser.add_argument(
        "--skip_execution",
        action="store_true",
        help=(
            "Skip tool execution (legacy mode). Produces empty result "
            "fields, matching the old script behaviour."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a query-config JSON file that maps molecule/reaction "
            "index ranges to query types.  When provided, entries are "
            "generated from the config instead of the hardcoded defaults.  "
            "See query_config.json for the expected schema."
        ),
    )
    args = parser.parse_args()

    # ---- load input data --------------------------------------------------
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "molecules" not in data or "reactions" not in data:
        parser.error(
            "Input file must be a JSON object with both "
            '"molecules" and "reactions" keys: '
            '{"molecules": [...], "reactions": [...]}'
        )
    molecules = data["molecules"]
    reactions: list[dict] = data["reactions"]

    execute = not args.skip_execution

    # ---- load query config (optional) -------------------------------------
    query_config: dict | None = None
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            query_config = json.load(f)
        log.info("Loaded query config from %s", args.config)

    # ---- generate ---------------------------------------------------------
    dataset = generate_ground_truth(
        molecules,
        reactions=reactions,
        execute=execute,
        config=query_config,
    )

    # ---- write output -----------------------------------------------------
    output_path = Path(args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"\nGenerated {len(dataset)} evaluation entries -> {output_path}")

    if execute:
        # Summarise success / failure counts.
        ok = 0
        for d in dataset:
            res = d["answer"]["result"]
            if isinstance(res, dict) and res.get("status") == "error":
                continue
            ok += 1
        print(f"  {ok}/{len(dataset)} entries executed successfully")


if __name__ == "__main__":
    main()
