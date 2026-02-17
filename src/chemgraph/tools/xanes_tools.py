
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import parsl
import shutil
from pathlib import Path
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from parsl import bash_app, File
from parsl.executors import HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.config import Config
from parsl.launchers import SingleNodeLauncher
from langchain_core.tools import tool

# API Configuration
MP_API_KEY = 'Pgvt9Q4pctLJeK7hDpB2F3ztUIjnDeym'

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def write_fdmnes_input(ase_atoms: Atoms,
                       z_absorber: int = None,
                       input_file_dir: Path = None,
                       radius: float = 6,
                       magnetism: bool = False,
                       ):
    if not isinstance(ase_atoms, Atoms):
        raise TypeError('ase_atoms must be an ase.Atoms object')

    atomic_numbers = ase_atoms.get_atomic_numbers()
    if z_absorber is None:
        z_absorber = atomic_numbers.max()

    if input_file_dir is None:
        input_file_dir = Path.cwd()

    with open(input_file_dir / 'fdmfile.txt', 'w') as f:
        f.write('1' + '\n')
        f.write('fdmnes_in.txt' + '\n')

    with open(input_file_dir / 'fdmnes_in.txt', 'w') as f:
        f.write('Filout' + '\n')
        f.write(f'{input_file_dir.name}' + 2*'\n')

        # Sets the energy mesh
        f.write('Range' + '\n')
        f.write('-55. 1.0 -10. 0.01 5. 0.1 150.' + 2*'\n')

        # Cluster Radius
        f.write('Radius' + '\n')
        f.write(f'{radius}' + 2*'\n')

        # Atomic number of the X-ray absorber
        f.write('Z_absorber' + '\n')
        f.write(f'{z_absorber}' + 2*'\n')

        # Enables magnetic contributions
        if magnetism:
            f.write('Magnetism' + 2*'\n')

        f.write('Green' + '\n')  # Use Green's function formalism for multiple scattering treatment
        f.write('Density_all' + '\n')  # Output total electron density for the cluster
        f.write('Quadrupole' + '\n')  # Include quadrupole transitions
        f.write('Spherical' + '\n')  # Start from spherical atomic densities
        f.write('SCF' + 2*'\n')  # Perform self-consistent field calculations on the cluster charge density

        if all(ase_atoms.pbc):
            f.write('Crystal' + '\n')

            # Writing cell lengths and angles
            f.write(' '.join(map(str, ase_atoms.cell.cellpar())) + '\n')

            # Atomic positions in fractional coordinates per required by periodic systems
            positions = np.round(ase_atoms.get_scaled_positions(), 6)

        else:
            f.write('Molecule' + '\n')

            # Writing cell lengths and angles
            cell_length = abs(ase_atoms.get_positions().max()) + abs(ase_atoms.get_positions().min())
            f.write(f'{cell_length} {cell_length} {cell_length} 90 90 90' + '\n')

            # Atomic positions in Cartesian coordinates per required by molecular systems
            positions = np.round(ase_atoms.get_positions(), 6)

        for i, position in enumerate(positions):
            f.write(f'{atomic_numbers[i]} ' + ' '.join(map(str, position)) + '\n')

        f.write('\n')
        f.write('Convolution' + '\n')
        f.write('End')


def get_normalized_xanes(conv_file: Path | str,
                         pre_edge_width: float = 20.0,
                         post_edge_width: float = 50.0,
                         calc_E0: bool = False
                         ) -> tuple[np.ndarray, np.ndarray]:
    energy_xas = np.loadtxt(conv_file, skiprows=1) # (N,2) array

    E = energy_xas[:, 0].astype(float)
    mu = energy_xas[:, 1].astype(float)

    # Finding edge energy E0 (onset of absorption) if file doesn't set 0 as reference
    if calc_E0:
        dmu_dE = np.gradient(mu, E)
        E0 = E[np.argmax(dmu_dE)]
    else:
        E0 = 0

    # Finding pre- and post-edge masks
    pre_mask = E <= (E0 - pre_edge_width)
    post_mask = E >= (E0 + post_edge_width)

    # Doing linear fits μ ~ m*E + b
    m_pre, b_pre = np.polyfit(E[pre_mask], mu[pre_mask], 1)
    m_post, b_post = np.polyfit(E[post_mask], mu[post_mask], 1)

    # Subtract pre-edge to shift the pre_line to mu = 0
    pre_line = m_pre*E + b_pre
    mu_corr = mu - pre_line

    # Computing normalized mu
    step = (m_post*E0 + b_post) - (m_pre*E0 + b_pre)
    mu_norm = mu_corr / step

    return np.column_stack([E, mu_norm]), energy_xas


def extract_conv(fdmnes_output_dir: Path | str) -> np.ndarray:
    if not isinstance(fdmnes_output_dir, Path):
        fdmnes_output_dir = Path(fdmnes_output_dir)

    energy_xas = {}
    for i, conv_file in enumerate(fdmnes_output_dir.glob('*conv.txt')):
        energy_xas[i] = np.loadtxt(conv_file, skiprows=1) # (N,2) array

    return energy_xas

# -----------------------------------------------------------------------------
# Workflow Steps
# -----------------------------------------------------------------------------

def fetch_materials_project_data(chemsys: list[str], db_path: Path):
    """Fetch materials data from Materials Project."""
    print(f"Fetching data from MP for: {chemsys}")
    atoms_list = []
    
    # Ensure correct API key usage
    with MPRester(MP_API_KEY) as mpr:
        doc_list = mpr.materials.summary.search(
            fields=['material_id', 'structure'], # 'xas', 'dos', 'symmetry'
            # has_props=['dos', 'xas'],
            energy_above_hull=(0, 0.001),
            formula=chemsys,
            deprecated=False,
            num_chunks=1,
            chunk_size=1,
        )

        for doc in doc_list:
            ase_atoms = AseAtomsAdaptor.get_atoms(doc.structure)
            ase_atoms.info.update({'MP-id' : str(doc.material_id),
                                #    'MP-xas': doc.xas,
                                #    'MP-dos': doc.dos
                                })
            atoms_list.append(ase_atoms)

    if not db_path.exists():
        db_path.mkdir(parents=True)
        
    with open(db_path / 'atoms_db.pkl', 'wb') as f:
        pickle.dump(atoms_list, f)
    
    print(f"Saved {len(atoms_list)} structures to {db_path / 'atoms_db.pkl'}")
    return atoms_list

def create_fdmnes_inputs(root_dir: Path):
    """Create FDMNES inputs from the database."""
    print("Creating FDMNES inputs...")
    runs_dir = root_dir / 'fdmnes_batch_runs'
    
    start_idx = 0
    if runs_dir.exists():
        for subdir in runs_dir.iterdir():
            try:
                start_idx = max(start_idx, int(subdir.name.split('_')[-1]))
            except ValueError:
                continue
        last_run = runs_dir / f'run_{start_idx}'
        if last_run.exists():
            shutil.rmtree(last_run)
    else:
        runs_dir.mkdir(parents=True)

    with open(root_dir / 'atoms_db.pkl', 'rb') as f:
        atoms_list = pickle.load(f)

    for i, atoms in enumerate(atoms_list, start=start_idx):
        curr_run_dir = runs_dir / f'run_{i}'
        curr_run_dir.mkdir(parents=True, exist_ok=True)

        z_absorber = max(atoms.get_atomic_numbers())
        write_fdmnes_input(ase_atoms=atoms,
                           input_file_dir=curr_run_dir,
                           z_absorber=z_absorber,
                           radius=6,
                           magnetism=False)

        pkl_filename = f'Z{z_absorber}_{atoms.info["MP-id"]}_{atoms.get_chemical_formula()}.pkl'
        with open(curr_run_dir / pkl_filename, 'wb') as f:
            pickle.dump(atoms, f)
            
    return runs_dir

def run_fdmnes_parsl_workflow(runs_dir: Path):
    """Run FDMNES calculations using Parsl."""
    print("Running Parsl workflow...")
    
    # Only run if we are in an environment that likely supports it or user asked specifically
    # Here we just implement the logic.
    
    # ---------- USER VARIABLES ----------
    account     = 'xanes_fmCatal' # account to charge
    num_nodes   = 2           # max number of nodes to use
    walltime    = '23:00:00'   # job length
    fdmnes_exe  = '/home/vferreiragrizzi/parallel_fdmnes/mpirun_fdmnes'
    num_cores   = int(os.environ.get('PBS_NP', '128'))
    # ----------------------------------------------
    
    def is_calc_done(run_dir: Path) -> bool:
        conv = next(run_dir.glob('*_conv.txt'), None)
        return conv is not None and conv.stat().st_size > 1024

    run_dirs = [d for d in runs_dir.glob('run_*') if d.is_dir() and not is_calc_done(d)]
    if not run_dirs:
        print('All calcs are done already!!')
        return

    @bash_app
    def run_fdmnes(run_dir, ncores, exe, stdout=None, stderr=None, outputs=None, cwd=None):
        return f"""
            cd "{run_dir}"
            "{exe}" -np {ncores}
            """

    try:
        htex = HighThroughputExecutor(
            label='htex',
            max_workers_per_node=1,
            cores_per_worker=num_cores,
            provider=PBSProProvider(
                account=account,
                nodes_per_block=1,
                cpus_per_node=num_cores,
                init_blocks=min(num_nodes, len(run_dirs)),
                max_blocks=num_nodes,
                walltime=walltime,
                scheduler_options='#PBS -N FDMNES_parsl',
                worker_init="""
                    source ~/miniconda/etc/profile.d/conda.sh
                    conda activate chemgraph
                    export OMP_NUM_THREADS=1
                """,
                launcher=SingleNodeLauncher(),
            ),
        )
        
        # Load parsl config if not already loaded
        try:
            parsl.load(Config(executors=[htex], retries=2))
        except RuntimeError:
            # Already loaded
            pass

        futures = []
        for curr_dir in run_dirs:
            expected_output_file = curr_dir / f"{curr_dir.name}_conv.txt"

            futures.append(
                run_fdmnes(
                    run_dir=str(curr_dir),
                    ncores=num_cores,
                    exe=fdmnes_exe,
                    cwd=str(curr_dir),
                    stdout=str(curr_dir / 'fdmnes_stdout.txt'),
                    stderr=str(curr_dir / 'fdmnes_stderr.txt'),
                    outputs=[File(str(expected_output_file))]
                )
            )

        print(f'Submitted {len(futures)} runs to Parsl')
        for fut in futures:
            fut.result()
        print('All runs finished')
        
    finally:
        parsl.clear()

def expand_database_results(root_dir: Path, runs_dir: Path):
    """Expand the database with XANES results."""
    print("Expanding database...")
    expanded_atoms_list = []
    for sub_dir in runs_dir.glob('run_*'):
        atoms_pkl_files = list(sub_dir.glob('*.pkl'))
        if not atoms_pkl_files:
            continue
            
        atoms_pkl_file = atoms_pkl_files[0]
        with open(atoms_pkl_file, 'rb') as f:
            ase_atoms = pickle.load(f)

        ase_atoms.info.update({'FDMNES-xanes': extract_conv(fdmnes_output_dir=sub_dir)})
        expanded_atoms_list.append(ase_atoms)

    with open(root_dir / 'atoms_db_expanded.pkl', 'wb') as f:
        pickle.dump(expanded_atoms_list, f)

def _plot_xanes_results_internal(root_dir: Path, runs_dir: Path):
    """Plot XANES results."""
    print("Plotting results...")
    # Naive plotting for now: plot whatever is found in the last few runs
    # This logic matches the original script's intent of plotting specific files, 
    # but here we generalize to plot valid results found in the run directory.
    
    for sub_dir in runs_dir.glob('run_*'):
        conv_file = next(sub_dir.glob('*_conv.txt'), None)
        if conv_file:
            try:
                norm_energy, energy = get_normalized_xanes(conv_file)
                plt.figure()
                plt.plot(norm_energy[:,0], norm_energy[:,1], label=sub_dir.name)
                plt.xlabel('Energy [eV]')
                plt.ylabel('Normalized Absorption')
                plt.title(f'XANES for {sub_dir.name}')
                plt.savefig(sub_dir / 'xanes_plot.png')
                plt.close()
                print(f"Plotted {sub_dir.name}")
            except Exception as e:
                print(f"Failed to plot {sub_dir.name}: {e}")

# -----------------------------------------------------------------------------
# Individual Workflow Tools
# -----------------------------------------------------------------------------

def _get_data_dir() -> Path:
    """Helper to determine the data directory."""
    cwd = Path.cwd()
    if 'PBS_O_WORKDIR' in os.environ:
         cwd = Path(os.environ['PBS_O_WORKDIR'])
    
    data_dir = cwd / 'xanes_data'
    if not data_dir.exists():
        data_dir.mkdir()
    return data_dir

@tool
def fetch_xanes_data(chemsys: list[str]) -> str:
    """
    Step 1: Fetch materials data from Materials Project for XANES workflow.
    
    Parameters:
    -----------
    chemsys : list[str]
        List of chemical systems to search for (e.g. ['Fe2O3', 'CoO'])
    """
    try:
        data_dir = _get_data_dir()
        atoms_list = fetch_materials_project_data(chemsys, data_dir)
        return f"Fetched {len(atoms_list)} structures for {chemsys} into {data_dir}"
    except Exception as e:
        return f"Error fetching data: {e}"

@tool
def create_xanes_inputs() -> str:
    """
    Step 2: Create FDMNES input files from the fetched database.
    Requires 'fetch_xanes_data' to have been run first.
    """
    try:
        data_dir = _get_data_dir()
        create_fdmnes_inputs(data_dir)
        return f"Created FDMNES inputs in {data_dir / 'fdmnes_batch_runs'}"
    except Exception as e:
        return f"Error creating inputs: {e}"

@tool
def run_xanes_parsl() -> str:
    """
    Step 3: Run FDMNES calculations using Parsl.
    Requires 'create_xanes_inputs' to have been run first.
    This may take a significant amount of time.
    """
    try:
        data_dir = _get_data_dir()
        runs_dir = data_dir / 'fdmnes_batch_runs'
        if not runs_dir.exists():
             return "Error: fdmnes_batch_runs directory not found. Did you run create_xanes_inputs?"
        
        run_fdmnes_parsl_workflow(runs_dir)
        return "Parsl execution finished."
    except Exception as e:
        return f"Error running Parsl: {e}"

@tool
def expand_xanes_db() -> str:
    """
    Step 4: Expand the database with calculation results.
    Requires 'run_xanes_parsl' to have completed.
    """
    try:
        data_dir = _get_data_dir()
        runs_dir = data_dir / 'fdmnes_batch_runs'
        expand_database_results(data_dir, runs_dir)
        return f"Database expanded with results in {data_dir}"
    except Exception as e:
        return f"Error expanding database: {e}"

@tool
def plot_xanes_results() -> str:
    """
    Step 5: Plot XANES results.
    Generates plots for completed calculations in the run directories.
    """
    try:
        data_dir = _get_data_dir()
        runs_dir = data_dir / 'fdmnes_batch_runs'
        plot_xanes_results(data_dir, runs_dir)
        return f"Plots generated in subdirectories of {runs_dir}"
    except Exception as e:
        return f"Error plotting results: {e}"

# -----------------------------------------------------------------------------
# Main Tool
# -----------------------------------------------------------------------------

@tool
def run_xanes_workflow(chemsys: list[str]) -> str:
    """
    Run the FULL XANES workflow for a given chemical system.
    
    This executes all steps sequentially:
    1. Fetching materials from Materials Project.
    2. Creating FDMNES input files.
    3. Running FDMNES calculations via Parsl.
    4. Expanding the database with results.
    5. Plotting the results.
    
    Parameters:
    -----------
    chemsys : list[str]
        List of chemical systems to search for (e.g. ['Fe2O3', 'CoO'])
        
    Returns:
    --------
    str
        Status message indicating completion or failure.
    """
    try:
        data_dir = _get_data_dir()
            
        print(f"Starting XANES workflow for {chemsys} in {data_dir}...")
        
        # 1. Fetch Data
        fetch_materials_project_data(chemsys, data_dir)
        
        # 2. Creates Inputs
        create_fdmnes_inputs(data_dir)
        
        # 3. Parsl Execution
        runs_dir = data_dir / 'fdmnes_batch_runs'
        run_fdmnes_parsl_workflow(runs_dir)
        
        # 4. Expand DB
        expand_database_results(data_dir, runs_dir)
        
        # 5. Plot
        _plot_xanes_results_internal(data_dir, runs_dir)
        
        return f"XANES workflow completed successfully for {chemsys}. Results in {data_dir}"
        
    except Exception as e:
        return f"Error executing XANES workflow: {str(e)}"
