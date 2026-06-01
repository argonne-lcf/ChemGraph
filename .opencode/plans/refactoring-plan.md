# ChemGraph Codebase Reorganization Plan

**Date:** 2025-04-25
**Status:** Completed (2026-04-25)

## Goal

Establish a consistent `*_core.py` + thin-wrapper pattern across all tool
domains, eliminate all code duplication, and consolidate schemas into the
`schemas/` package. After this refactoring every domain follows:

```
schemas/<domain>_schema.py           # Pydantic input/output models
tools/<domain>_core.py               # Pure-Python logic (no decorators)
tools/<domain>_tools.py              # LangChain @tool wrappers (thin)
mcp/<domain>_mcp.py or *_parsl.py    # MCP @mcp.tool wrappers (thin)
```

---

## Context: What Was Already Done

In the first refactoring pass we unified the ASE simulation logic:

- Created `tools/core.py` — single `run_ase_core()` + shared helpers
  (`atoms_to_atomsdata`, `is_linear_molecule`, `get_symmetry_number`,
  `load_calculator`, `_resolve_path`, etc.)
- Simplified `tools/ase_tools.py` → thin `@tool` wrappers (737 → 187 lines)
- Simplified `mcp/mcp_tools.py` → thin `@mcp.tool` wrappers (554 → 206 lines)
- Simplified `tools/parsl_tools.py` → `run_mace_core()` delegates to
  `run_ase_core()` via schema conversion (406 → 204 lines)
- Reduced `tools/mcp_helper.py` to a backward-compat re-export shim (158 → 13 lines)

All 19 relevant tests pass. The 59 failures in the full suite are pre-existing
(missing `vision_test_graph`, `langchain_huggingface`, `architector_tools`,
`_detect_executor_failure`).

---

## Current File Inventory (with line counts)

### tools/
| File                       | Lines | Status                                    |
|----------------------------|------:|-------------------------------------------|
| `core.py`                  |   611 | To rename → `ase_core.py`                 |
| `ase_tools.py`             |   187 | Already refactored (thin wrappers)        |
| `cheminformatics_tools.py` |   152 | Needs core extraction                     |
| `graspa_tools.py`          |   307 | Has core+wrapper in one file              |
| `xanes_tools.py`           |   616 | Has core+wrapper in one file              |
| `parsl_tools.py`           |   204 | Schemas need to move to `schemas/`        |
| `report_tools.py`          |   776 | Has duplicated element_map (lines 347,415)|
| `mcp_helper.py`            |    13 | Delete (re-export shim)                   |
| `rag_tools.py`             |   343 | No changes needed                         |
| `generic_tools.py`         |   101 | No changes needed                         |

### mcp/
| File                    | Lines | Status                                    |
|-------------------------|------:|-------------------------------------------|
| `mcp_tools.py`          |   206 | Needs cheminformatics core extraction     |
| `mace_mcp_parsl.py`     |   256 | Update schema import path                 |
| `graspa_mcp_parsl.py`   |   201 | Extract `load_parsl_config` to server_utils|
| `xanes_mcp_parsl.py`    |   290 | Merge with `xanes_mcp.py`, extract config |
| `xanes_mcp.py`          |    97 | Delete (merge into `xanes_mcp_parsl.py`)  |
| `data_analysis_mcp.py`  |   304 | No changes needed                         |
| `server_utils.py`       |    79 | Add `load_parsl_config()`                 |

### schemas/
| File                    | Status                                      |
|-------------------------|---------------------------------------------|
| `ase_input.py`          | No changes                                  |
| `atomsdata.py`          | No changes                                  |
| `graspa_schema.py`      | No changes                                  |
| `graspa_input.py`       | No changes (separate schema, used elsewhere)|
| `xanes_schema.py`       | No changes                                  |
| `agent_response.py`     | No changes                                  |
| `multi_agent_response.py`| No changes                                 |
| `calculators/`          | Add `__init__.py`                           |
| **`mace_parsl_schema.py`** | **NEW** — moved from `parsl_tools.py`    |

---

## Phases

### Phase 1: Rename and Relocate (mechanical, low risk)

#### 1A. Rename `tools/core.py` → `tools/ase_core.py`

**Why:** The name `core.py` is ambiguous once we add `cheminformatics_core.py`,
`graspa_core.py`, `xanes_core.py`.

**Files to update (imports):**
- `tools/ase_tools.py` — `from chemgraph.tools.core import …` → `from chemgraph.tools.ase_core import …`
- `tools/mcp_helper.py` — same (will be deleted in Phase 4A anyway, but update
  first so we can delete cleanly)
- `tools/parsl_tools.py` — `from chemgraph.tools.core import run_ase_core` → `from chemgraph.tools.ase_core import run_ase_core`
- `mcp/mcp_tools.py` — `from chemgraph.tools.core import …` → `from chemgraph.tools.ase_core import …`

No other files import from `tools/core.py` directly.

**Verification:** `pytest tests/test_tools.py tests/test_mace.py tests/test_mcp.py tests/test_calculators.py -v`

#### 1B. Move MACE schemas to `schemas/mace_parsl_schema.py`

**What moves:**
- `mace_input_schema` (class) — from `parsl_tools.py`
- `mace_input_schema_ensemble` (class) — from `parsl_tools.py`
- `mace_output_schema` (class) — from `parsl_tools.py`

**Files to update (imports):**
- `tools/parsl_tools.py` — add `from chemgraph.schemas.mace_parsl_schema import mace_input_schema, mace_output_schema`
- `mcp/mace_mcp_parsl.py` line 15-19 — `from chemgraph.tools.parsl_tools import (mace_input_schema, mace_input_schema_ensemble, run_mace_core)` → split into:
  - `from chemgraph.schemas.mace_parsl_schema import mace_input_schema, mace_input_schema_ensemble`
  - `from chemgraph.tools.parsl_tools import run_mace_core`
- `mcp/mace_mcp_parsl.py` line 39 (deferred import inside `run_mace_parsl_app`) — update similarly

**Verification:** `pytest tests/test_mace.py tests/test_mcp.py -v`

#### 1C. Add missing `__init__.py` files

Create empty `__init__.py` in:
- `src/chemgraph/mcp/__init__.py`
- `src/chemgraph/schemas/calculators/__init__.py`

These are currently implicit namespace packages. Adding `__init__.py` makes them
consistent with `tools/` and `utils/`.

---

### Phase 2: Extract Core Modules (eliminates duplication)

#### 2A. Create `tools/cheminformatics_core.py`

**Currently duplicated logic:**
- RDKit SMILES→3D pipeline appears **3 times**:
  - `cheminformatics_tools.py:smiles_to_atomsdata()` (lines 37-83)
  - `cheminformatics_tools.py:smiles_to_coordinate_file()` (lines 86-152)
  - `mcp/mcp_tools.py:smiles_to_coordinate_file()` (lines 82-159)
- PubChem name→SMILES lookup appears **2 times** with **different return types**:
  - `cheminformatics_tools.py:molecule_name_to_smiles()` → returns `{"name": ..., "smiles": ...}`
  - `mcp/mcp_tools.py:molecule_name_to_smiles()` → returns just the SMILES string

**New file: `tools/cheminformatics_core.py`**
```python
def smiles_to_3d(smiles: str, seed: int = 2025) -> tuple[list[int], list[list[float]]]:
    """SMILES → (atomic_numbers, positions) via RDKit. Single implementation."""

def molecule_name_to_smiles_core(name: str) -> str:
    """PubChem name → canonical SMILES. Returns the raw SMILES string."""

def smiles_to_coordinate_file_core(smiles: str, output_file: str, seed: int) -> dict:
    """SMILES → coordinate file on disk. Returns result dict."""

def smiles_to_atomsdata_core(smiles: str, seed: int) -> AtomsData:
    """SMILES → AtomsData object."""
```

**Files to simplify:**
- `cheminformatics_tools.py` → thin `@tool` wrappers calling `cheminformatics_core.*`
  - `molecule_name_to_smiles` wraps `molecule_name_to_smiles_core`, adds the dict return
  - `smiles_to_atomsdata` wraps `smiles_to_atomsdata_core`
  - `smiles_to_coordinate_file` wraps `smiles_to_coordinate_file_core`
- `mcp/mcp_tools.py` → `molecule_name_to_smiles` and `smiles_to_coordinate_file` become
  thin `@mcp.tool` wrappers calling `cheminformatics_core.*`
  - Remove `_resolve_path` local definition (use from `ase_core` or `cheminformatics_core`)

**Verification:** `pytest tests/test_tools.py tests/test_mcp.py -v -k "smiles or molecule"`

#### 2B. Create `tools/xanes_core.py` (higher priority — full file duplication)

**Currently duplicated:**
- `xanes_mcp.py` is an exact subset of `xanes_mcp_parsl.py` — `run_xanes_single`,
  `fetch_mp_structures`, `plot_xanes` are identical in both files.
- Both use default port 9007 (would conflict if run simultaneously).

**New file: `tools/xanes_core.py`**

Extract from `xanes_tools.py`:
```python
def write_fdmnes_input(...): ...
def get_normalized_xanes(...): ...
def extract_conv(...): ...
def _get_data_dir(): ...
def run_xanes_core(params): ...
def fetch_materials_project_data(params, db_path): ...
def create_fdmnes_inputs(...): ...
def expand_database_results(...): ...
def plot_xanes_results(...): ...
```

**Files to simplify:**
- `xanes_tools.py` → thin `@tool` wrappers:
  - `run_xanes` wraps `run_xanes_core`
  - `fetch_xanes_data` wraps `fetch_materials_project_data`
  - `plot_xanes_data` wraps `plot_xanes_results`
- `xanes_mcp_parsl.py` → imports from `xanes_core` instead of deferred imports
  from `xanes_tools`
- **Delete `xanes_mcp.py`** — merge its tools into `xanes_mcp_parsl.py` with
  conditional Parsl support (only expose `run_xanes_ensemble` if Parsl is loaded)

**Verification:** Manual (XANES tests require FDMNES executable which is HPC-only)

#### 2C. Create `tools/graspa_core.py` (lower priority)

**Current state:** `graspa_tools.py` already has `run_graspa_core()` as a plain
function with `@tool run_graspa()` as the wrapper — the pattern is there, just
in a single file.

**New file: `tools/graspa_core.py`**

Extract from `graspa_tools.py`:
```python
def _read_graspa_sycl_output(output_dir): ...
def mock_graspa(): ...
def run_graspa_core(params: graspa_input_schema): ...
```

**Files to simplify:**
- `graspa_tools.py` → only `@tool run_graspa()` wrapper
- `graspa_mcp_parsl.py` → deferred import changes from
  `from chemgraph.tools.graspa_tools import run_graspa_core` to
  `from chemgraph.tools.graspa_core import run_graspa_core`

**Verification:** Manual (gRASPA requires SYCL runtime)

---

### Phase 3: Consolidate Shared Utilities

#### 3A. Move `extract_output_json` to `ase_core.py`

**Currently duplicated 3 times** (identical `json.load` logic):
- `tools/ase_tools.py` — `@tool extract_output_json`
- `mcp/mcp_tools.py` — `@mcp.tool extract_output_json`
- `mcp/mace_mcp_parsl.py` — `@mcp.tool extract_output_json`

**Action:**
- Add `extract_output_json_core(json_file: str) -> dict` to `tools/ase_core.py`
- All three wrappers call it

#### 3B. Consolidate `load_parsl_config()` into `mcp/server_utils.py`

**Currently duplicated 2 times** (identical function):
- `mcp/graspa_mcp_parsl.py` lines 44-66
- `mcp/xanes_mcp_parsl.py` lines 39-65

**Action:**
- Add `load_parsl_config(compute_system: str) -> Config` to `mcp/server_utils.py`
- Both MCP servers import from there

#### 3C. Fix `report_tools.py` duplicated element map

**Currently:** Two identical `element_map` dicts (lines 347 and 415) covering
only H(1) through Ar(18). Heavier elements fall through to `X{num}`.

**Action:**
- Replace with `from ase.data import chemical_symbols`
- Use `chemical_symbols[num]` instead of `element_map.get(num, f"X{num}")`
- Handles all elements automatically

---

### Phase 4: Cleanup

#### 4A. Delete `tools/mcp_helper.py`

**Current state:** 13-line re-export shim. Only one remaining consumer imports
from `mcp_helper` instead of `ase_core`:
- `tools/cheminformatics_tools.py` line 10: `from chemgraph.tools.mcp_helper import _resolve_path`

The `new_eval/scripts/mcp_example/` files also import from `mcp_helper`, but
those are separate example scripts (see 4B).

**Action:**
1. Update `cheminformatics_tools.py` to import `_resolve_path` from
   `chemgraph.tools.ase_core` (or from the new `cheminformatics_core.py` if
   Phase 2A is done first)
2. Delete `tools/mcp_helper.py`

#### 4B. Refactor `new_eval/scripts/mcp_example/` scripts

**Currently:** Both `mcp_http/start_mcp_server.py` and
`mcp_stdio/mcp_tools_stdio.py` contain **inline copies** of `run_ase()`
(~200+ lines each) plus import helpers from `mcp_helper`.

**Action:**
- Replace inline `run_ase` with `from chemgraph.tools.ase_core import run_ase_core`
- Update helper imports from `mcp_helper` → `ase_core`
- Each file shrinks from ~200+ lines of duplicated simulation logic to ~20 lines

#### 4C. Fix stale import in `utils/tool_call_eval.py`

**Line 4:** `from chemgraph.models.ase_input import ASEInputSchema`
**Should be:** `from chemgraph.schemas.ase_input import ASEInputSchema`

(The `chemgraph.models` path does not contain `ase_input` — this is either dead
code or a bug.)

---

## Execution Order (with dependencies)

```
Phase 1A: Rename core.py → ase_core.py
    └── Phase 1B: Move MACE schemas to schemas/mace_parsl_schema.py
    └── Phase 1C: Add missing __init__.py files
    └── Phase 4A: Delete mcp_helper.py
         └── Phase 2A: Create cheminformatics_core.py
              └── Phase 3A: Move extract_output_json to ase_core.py
    └── Phase 2B: Create xanes_core.py + merge xanes_mcp.py
         └── Phase 3B: Consolidate load_parsl_config to server_utils.py
    └── Phase 2C: Create graspa_core.py (optional, lower priority)
    └── Phase 3C: Fix report_tools.py element map
    └── Phase 4B: Refactor new_eval example scripts
    └── Phase 4C: Fix stale import in tool_call_eval.py
```

Suggested implementation order (each step leaves the repo in a passing state):

1. **Phase 1A** — rename `core.py` → `ase_core.py`, update 4 import sites
2. **Phase 1B** — move MACE schemas, update 2 files
3. **Phase 1C** — add 2 empty `__init__.py` files
4. **Phase 4A** — fix 1 import in `cheminformatics_tools.py`, delete `mcp_helper.py`
5. **Phase 3A** — add `extract_output_json_core` to `ase_core.py`, update 3 wrappers
6. **Phase 2A** — create `cheminformatics_core.py`, simplify 2 files
7. **Phase 3C** — fix `report_tools.py` element map (2 occurrences)
8. **Phase 4C** — fix stale import (1 line)
9. **Phase 2B** — create `xanes_core.py`, merge `xanes_mcp.py` into `xanes_mcp_parsl.py`
10. **Phase 3B** — extract `load_parsl_config` to `server_utils.py`
11. **Phase 2C** — create `graspa_core.py` (optional)
12. **Phase 4B** — refactor `new_eval` example scripts

---

## Target File Structure (after all phases)

```
src/chemgraph/
  schemas/
    __init__.py
    ase_input.py                     # ASEInputSchema, ASEOutputSchema
    atomsdata.py                     # AtomsData
    graspa_schema.py                 # graspa_input_schema, graspa_input_schema_ensemble
    graspa_input.py                  # GRASPAInputSchema (separate, pre-existing)
    xanes_schema.py                  # xanes_input_schema, xanes_input_schema_ensemble
    mace_parsl_schema.py             # NEW: mace_input/output/ensemble schemas
    agent_response.py                # ResponseFormatter, etc.
    multi_agent_response.py          # MultiAgentResponse
    calculators/
      __init__.py                    # NEW (empty)
      mace_calc.py
      emt_calc.py
      tblite_calc.py
      orca_calc.py
      nwchem_calc.py
      fairchem_calc.py
      aimnet2_calc.py
      mopac_calc.py
      psi4_calc.py

  tools/
    __init__.py
    ase_core.py                      # RENAMED from core.py
    ase_tools.py                     # @tool wrappers → ase_core
    cheminformatics_core.py          # NEW: smiles_to_3d, name→SMILES, etc.
    cheminformatics_tools.py         # @tool wrappers → cheminformatics_core
    graspa_core.py                   # NEW: run_graspa_core extracted
    graspa_tools.py                  # @tool wrapper → graspa_core
    xanes_core.py                    # NEW: run_xanes_core, helpers extracted
    xanes_tools.py                   # @tool wrappers → xanes_core
    parsl_tools.py                   # run_mace_core (schemas imported from schemas/)
    report_tools.py                  # fixed element_map → ase.data.chemical_symbols
    rag_tools.py                     # unchanged
    generic_tools.py                 # unchanged
    # mcp_helper.py                  # DELETED

  mcp/
    __init__.py                      # NEW (empty)
    server_utils.py                  # + load_parsl_config() consolidated
    mcp_tools.py                     # @mcp.tool → ase_core + cheminformatics_core
    mace_mcp_parsl.py                # schema import updated
    graspa_mcp_parsl.py              # → graspa_core, config from server_utils
    xanes_mcp_parsl.py               # → xanes_core, absorbed xanes_mcp.py
    # xanes_mcp.py                   # DELETED (merged into xanes_mcp_parsl.py)
    data_analysis_mcp.py             # unchanged

  utils/
    __init__.py
    async_utils.py                   # unchanged
    config_utils.py                  # unchanged
    get_workflow_from_llm.py         # unchanged
    logging_config.py                # unchanged
    parsing.py                       # unchanged
    tool_call_eval.py                # fixed stale import
```

---

## Test Commands

After each phase, run the relevant subset:

```bash
# Core ASE tests (phases 1A, 3A, 4A)
pytest tests/test_tools.py tests/test_mace.py tests/test_mcp.py tests/test_calculators.py -v

# After phase 2A (cheminformatics)
pytest tests/test_tools.py tests/test_mcp.py -v -k "smiles or molecule"

# Full suite (excluding pre-existing failures)
pytest tests/ -v \
  --ignore=tests/test_architector.py \
  --ignore=tests/test_multi_agent_retry.py \
  -k "not test_real_new_evaluation_ground_truth"
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking `new_eval/` scripts | Phase 4B is last; scripts are not part of CI |
| Parsl deferred imports break after rename | Update the deferred import inside `run_mace_parsl_app` in Phase 1B |
| `xanes_mcp.py` deletion breaks HPC users | Phase 2B merges functionality into `xanes_mcp_parsl.py` with conditional Parsl |
| Circular imports from `ase_core.py` | `ase_core.py` has no LangChain/MCP deps — no risk |
| Calculator schemas missing `__init__.py` | Phase 1C — adding empty file, no functional change |
