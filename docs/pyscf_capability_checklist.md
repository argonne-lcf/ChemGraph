# PySCF Capability Checklist

This checklist tracks PySCF support by capability family, not by individual
example file. The PySCF examples directory is useful as a discovery corpus, but
ChemGraph should expose stable, parameterized tools and recipes rather than one
MCP tool per example.

Current reference corpus:

- PySCF examples path: `/Users/jinchuli/projects/pyscf/examples`
- Python example count observed: 438
- Conservative direct v0 coverage estimate: about 12 / 438 examples

## Status Legend

| Status | Meaning |
|---|---|
| done | Implemented and covered by focused tests. |
| partial | Implemented for a narrow/common case; more schema, tests, or validation needed. |
| planned | Recommended future capability, but not implemented yet. |
| absent | Not implemented. |
| blocked | Needs domain decision, dependency, or external software before implementation. |

## Current MCP Surface

| MCP tool | Status | Purpose |
|---|---|---|
| `get_pyscf_capability_manifest` | done | Return the curated PySCF capability manifest exposed by ChemGraph. |
| `run_pyscf_molecular` | partial | Run molecular HF/DFT, optional MP2/CCSD/CCSD(T), and basic properties. |
| `run_pyscf_periodic` | partial | Run minimal periodic gamma-point or k-point HF/DFT. |
| `run_pyscf_property` | done | Extract stored properties from a ChemGraph PySCF JSON artifact. |
| `run_pyscf_recipe` | partial | Run whitelisted advanced recipes; currently only `casscf_single_point`. |
| `extract_pyscf_output` | done | Load a saved ChemGraph PySCF JSON artifact. |

## Capability Matrix

| Capability ID | Family | User intent | PySCF modules | Example paths | Schema status | Implementation status | Test status | Notes |
|---|---|---|---|---|---|---|---|---|
| `mol_structure_input` | input | Provide molecule by atom string or structure file. | `pyscf.gto`, ASE reader | `gto/00-input_mole.py`, `gto/01-input_geometry.py` | partial | partial | partial | Current schema supports atom strings and ASE-readable files, but not all Mole options. |
| `mol_charge_spin_basis` | input | Set charge, spin, basis, coordinate unit. | `pyscf.gto` | `gto/04-input_basis.py`, `scf/02-rohf_uhf.py` | partial | partial | partial | Current support handles common scalar fields, not custom parsed basis/ECP dictionaries. |
| `mol_scf_rhf` | molecular SCF | Run closed-shell RHF single point. | `pyscf.scf` | `scf/00-simple_hf.py`, `scf/01-h2o.py` | done | done | done | Good v0 smoke capability. |
| `mol_scf_uhf_rohf` | molecular SCF | Run open-shell UHF or ROHF single point. | `pyscf.scf` | `scf/02-rohf_uhf.py` | done | partial | partial | Needs more open-shell reference tests. |
| `mol_scf_controls` | molecular SCF | Control max cycles, convergence tolerance, memory, verbosity. | `pyscf.scf` | `scf/03-level_shift.py`, `scf/55-overload_convergence_criteria.py` | partial | partial | partial | Current schema has `max_cycle`, `conv_tol`, `max_memory`, `verbose`; no level shift, DIIS tuning, Newton solver, callbacks, or stability checks. |
| `mol_dft_rks_uks` | molecular DFT | Run RKS/UKS with an XC functional. | `pyscf.dft` | `dft/00-simple_dft.py`, `dft/13-rsh_dft.py` | partial | partial | partial | Current schema accepts `xc`; no grid controls, dispersion, DFT+U, custom XC backend, or noncollinear DFT. |
| `mol_dft_grid_controls` | molecular DFT | Control integration grids and grid response. | `pyscf.dft` | `dft/11-grid_scheme.py`, `grad/02-dft_grad.py` | absent | absent | absent | Needed for more reliable DFT gradients and advanced DFT examples. |
| `post_hf_mp2` | post-HF | Run MP2 after SCF. | `pyscf.mp` | `mp/00-simple_mp2.py` | done | done | done | Basic canonical MP2 only; no density-fitted, noncanonical, frozen-core, or natural orbital options. |
| `post_hf_ccsd` | post-HF | Run CCSD after SCF. | `pyscf.cc` | `cc/00-simple_ccsd.py` | done | done | partial | Basic CCSD only; no frozen core, density fitting, restart, EOM, lambda, or custom Hamiltonian options. |
| `post_hf_ccsdt` | post-HF | Run perturbative triples after CCSD. | `pyscf.cc` | `cc/00-simple_ccsd_t.py` | done | done | partial | Needs reference tests and clarity on restricted/unrestricted behavior. |
| `properties_dipole_population_mo` | properties | Return dipole, Mulliken population, and MO energies. | `pyscf.scf` | `scf/00-simple_hf.py`, `dft/00-simple_dft.py` | done | done | done | Current output is JSON serializable but not a complete PySCF analysis dump. |
| `properties_gradient` | properties | Return nuclear gradient. | `pyscf.grad` | `grad/01-scf_grad.py`, `grad/02-dft_grad.py` | partial | partial | partial | Basic `nuc_grad_method().kernel()` only; no atom subset, grid response, excited-state gradients, or post-HF gradients. |
| `properties_density_orbitals` | properties | Export density, orbitals, cube, Molden, or FCIDUMP artifacts. | `pyscf.tools`, `pyscf.fci` | `tools/01-fcidump.py`, `tools/02-molden.py`, `tools/05-cubegen.py` | absent | absent | absent | Strong next artifact/provenance target. |
| `artifact_json_chkfile` | artifacts | Save machine-readable results and checkpoint. | `json`, PySCF chkfile | `misc/02-chkfile.py`, `scf/14-restart.py` | partial | partial | done | JSON plus checkpoint exist; restart from checkpoint is not exposed. |
| `artifact_property_extract` | artifacts | Extract stored properties from prior result JSON. | ChemGraph JSON | ChemGraph-specific | done | done | done | This is ChemGraph artifact functionality, not a native PySCF example. |
| `artifact_output_extract` | artifacts | Load saved PySCF JSON output. | ChemGraph JSON | ChemGraph-specific | done | done | done | Useful for chained workflows and notebook benchmarks. |
| `pbc_cell_input` | periodic | Provide lattice vectors, basis, pseudo, charge, spin. | `pyscf.pbc.gto` | `pbc/00-input_cell.py`, `pbc/01-input_output_geometry.py`, `pbc/04-input_basis.py`, `pbc/05-input_pp.py` | partial | partial | partial | Current schema supports common scalar basis/pseudo values, not mixed dictionaries, ECP, fractional coordinates, or file round trips. |
| `pbc_gamma_scf_dft` | periodic | Run gamma-point periodic HF/DFT. | `pyscf.pbc.scf`, `pyscf.pbc.dft` | `pbc/10-gamma_point_scf.py` | partial | partial | partial | Minimal smoke support only. |
| `pbc_kpoint_scf_dft` | periodic | Run k-point periodic HF/DFT. | `pyscf.pbc.scf`, `pyscf.pbc.dft` | `pbc/20-k_points_scf.py` | partial | partial | partial | Supports k-mesh construction for K references; no density fitting, exxdiv, band structure, or symmetry controls. |
| `pbc_post_hf` | periodic | Run periodic MP2, CCSD, TDDFT, GW, ADC. | `pyscf.pbc.mp`, `pyscf.pbc.cc`, `pyscf.pbc.tdscf` | `pbc/12-gamma_point_post_hf.py`, `pbc/22-k_points_mp2.py`, `pbc/22-k_points_ccsd.py` | absent | absent | absent | Should remain separate from minimal PBC SCF until reference tests exist. |
| `mcscf_casscf_single_point` | multireference | Run a basic CASSCF single point with `ncas` and `nelecas`. | `pyscf.mcscf` | `mcscf/00-simple_casscf.py` | partial | partial | done | Implemented as recipe `casscf_single_point`; active-space choice remains user/domain responsibility. |
| `mcscf_casci_single_point` | multireference | Run CASCI after SCF. | `pyscf.mcscf` | `mcscf/00-simple_casci.py` | absent | absent | absent | Natural next recipe. |
| `mcscf_active_space_selection` | multireference | Select or reorder active orbitals. | `pyscf.mcscf` | `mcscf/10-define_cas_space.py`, `mcscf/43-avas.py`, `mcscf/43-dmet_cas.py` | absent | absent | absent | Needs domain-aware schema; poor defaults can produce misleading results. |
| `mcscf_state_average` | multireference | Run state-specific or state-averaged MCSCF. | `pyscf.mcscf` | `mcscf/15-state_average.py`, `mcscf/41-state_average.py` | absent | absent | absent | Good recipe candidate after basic CASCI/CASSCF tests. |
| `mrpt_nevpt2` | multireference | Run NEVPT2 or related MRPT methods. | `pyscf.mrpt` | `mrpt/03-df-nevpt2.py`, `mrpt/41-for_state_average.py` | absent | absent | absent | Requires validated multireference setup first. |
| `fci_basic` | exact diagonalization | Run FCI or selected CI. | `pyscf.fci` | `fci/00-simple_fci.py`, `fci/02-selected_ci.py` | absent | absent | absent | Could be a recipe family; size limits are important. |
| `td_tddft_basic` | excited states | Run TDHF/TDDFT/TDA and report excitation energies. | `pyscf.tdscf` | `tddft/00-simple_tddft.py`, `tddft/01-nto_analysis.py` | absent | absent | absent | Needs output schema for states, oscillator strengths, NTOs, and convergence. |
| `geomopt_basic` | geometry optimization | Optimize molecular geometry. | `pyscf.geomopt`, ASE, geomeTRIC, PyBerny | `geomopt/01-geomeTRIC.py`, `geomopt/01-ase.py`, `geomopt/01-pyberny.py` | absent | absent | absent | Needs dependency decisions and artifact schema for optimized structures and trajectory. |
| `hessian_frequencies_thermo` | vibrational analysis | Compute Hessian, frequencies, thermochemistry. | `pyscf.hessian`, `pyscf.hessian.thermo` | `hessian/01-scf_hessian.py`, `hessian/10-thermochemistry.py` | absent | absent | absent | Strong workflow-template target after geometry optimization exists. |
| `solvent_ddcosmo_pcm_smd` | environment | Run implicit solvent calculations. | `pyscf.solvent` | `solvent/00-scf_with_ddcosmo.py`, `solvent/05-pcm.py`, `solvent/06-smd.py` | absent | absent | absent | Current `solvent` field is reserved and rejected. |
| `qmmm_basic` | environment | Run QM/MM point charge embedding. | `pyscf.qmmm` | `qmmm/00-hf.py`, `qmmm/01-dft.py`, `qmmm/20-grad.py` | absent | absent | absent | Needs schema for MM charges, units, and force outputs. |
| `density_fitting` | acceleration | Enable density fitting or auxiliary basis controls. | `pyscf.df`, method `.density_fit()` | `df/00-with_df.py`, `scf/20-density_fitting.py`, `mp/10-dfmp2.py`, `cc/21-dfccsd.py` | absent | absent | absent | Useful because many PySCF examples and real jobs use DF variants. |
| `relativistic_x2c` | relativistic | Enable X2C or relativistic corrections. | `pyscf.x2c`, method `.x2c()` | `x2c/01-x2c1e.py`, `scf/21-x2c.py` | absent | absent | absent | Should be explicit because it changes physics and interpretation. |
| `ecp_custom_basis` | input | Use ECPs, parsed basis sets, mixed basis dictionaries. | `pyscf.gto`, `pyscf.pbc.gto` | `gto/05-input_ecp.py`, `pbc/05-input_pp.py` | absent | absent | absent | Important for heavier elements and periodic workflows. |
| `scans_pes` | workflow | Run bond scans or parameter sweeps. | PySCF plus workflow loop | `scf/16-h2_scan.py`, `scf/30-scan_pes.py`, `grad/16-scan_force.py` | absent | absent | absent | Better as a workflow/recipe layer than a single calculation field. |
| `restart_checkpoint` | workflow | Restart from checkpoint or reuse prior orbitals. | `pyscf.lib.chkfile`, SCF chkfile | `scf/14-restart.py`, `mcscf/13-restart.py`, `cc/32-restart.py` | absent | absent | absent | Needs provenance and compatibility checks. |
| `batch_many_jobs` | workflow | Run many independent calculations. | ChemGraph orchestration | benchmark/scans examples | absent | absent | absent | Could be implemented outside PySCF core tools as graph/batch execution. |
| `md_basic` | dynamics | Run molecular dynamics. | `pyscf.md` | `md/00-simple_nve.py`, `md/04-mb_nvt_berendson.py` | absent | absent | absent | Likely not a near-term MCP surface unless requested. |
| `eph_basic` | advanced properties | Electron-phonon coupling. | `pyscf.eph` | `eph/00-simple_eph.py` | absent | absent | absent | Advanced, domain-heavy. |
| `nmr_basic` | advanced properties | NMR calculations. | PySCF NMR modules | `nmr/crco6-nr-msc.py` | absent | absent | absent | Advanced, domain-heavy. |
| `orbital_localization` | analysis | Localize or analyze orbitals. | `pyscf.lo` | `local_orb/01-pop_with_nao.py`, `local_orb/07-pipek_mezey.py` | absent | absent | absent | Useful post-processing family after density/orbital artifacts exist. |
| `advanced_many_body` | advanced methods | Run ADC, GW, AGF2, EOM, MCPDFT. | `pyscf.adc`, `pyscf.gw`, `pyscf.agf2`, `pyscf.mcpdft` | `adc/01-closed_shell.py`, `gw/00-simple_gw.py`, `agf2/00-simple_agf2.py`, `mcpdft/00-simple_mcpdft.py` | absent | absent | absent | Keep as explicit recipes with strong tests, not generic parameters. |
| `external_solvers` | external | Use DMRG, SHCI, or other external solvers. | PySCF external solver interfaces | `mcscf/50-dmrgscf_with_block.py`, `mcscf/50-cornell_shci_casscf.py` | blocked | blocked | absent | Requires external binaries and environment-specific configuration. |
