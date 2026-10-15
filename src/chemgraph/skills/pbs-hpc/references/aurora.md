# Aurora

Use the [official job guide](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)
and `qstat -Qf QUEUE` for current queue limits. Preserve supplied project, queue,
walltime and resource choices for prepare-only work. Aurora uses Intel GPUs;
confirm that the requested calculator supports the intended CPU or Intel GPU
execution mode before choosing a launch command.

## Prepare the calculation environment

- Obtain the shared project directory from deployment configuration, typically
  under `/lus/flare/projects/PROJECT`. Declare the filesystems the job needs,
  such as `flare`; `qstat -Bf` exposes `resources_available.valid_filesystems`.
  Stage inputs and model weights where all workers can read them and write
  persistent results to shared storage.
- Use the site's compatible oneAPI, MPI and Python environment. The
  [Python guide](https://docs.alcf.anl.gov/aurora/data-science/python/) describes
  `module load frameworks` and extending it with a virtual environment. Load
  the base module before activating an environment built on it. Check the
  calculator's actual device support; CUDA settings do not enable Intel GPUs.
- Check `command -v mpiexec` after activating Python. MPI applications must use
  the site's PALS launcher and a compatible MPI library; an environment's own
  launcher can shadow PALS. Verify the MPI library used by `mpi4py` when needed.
- Perform calculator initialization and device checks on allocated compute
  nodes. Prepare-only syntax checks cannot establish runtime compatibility.

## Launch and GPU checks

Use the parent skill's [PBS template](../assets/job.pbs.template) for direct PBS
or the `iri-hpc` staging workflow for IRI. Both use the
[allocation helper](../assets/pbs-launch.sh).
Obtain inputs, the runner and result expectations from
[ASE calculations](../../chemgraph/references/ase-calculations.md).

- Launch a single calculation with the application's supported thread/device
  settings. Use `mpiexec` only for applications or worker launchers that need it.
  Starting multiple copies of the ASE runner with the same input and output
  paths does not distribute a calculation.
- Inspect `ZE_FLAT_DEVICE_HIERARCHY` after environment setup. `FLAT` exposes
  twelve tiles as devices; `COMPOSITE` exposes six GPUs with two tiles each.
  The `frameworks` environment uses `FLAT`.
- `ZE_AFFINITY_MASK` restricts visibility per process. A shared mask does not
  assign a different GPU to each rank. Preserve worker/application assignments
  and confirm each process's selected device before scaling up.
- Follow the [binding examples](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#binding-mpi-ranks-to-gpus)
  for the active hierarchy. Do not use the COMPOSITE `gpu_tile_compact.sh`
  wrapper unchanged in FLAT mode. Validate CPU binding and threads against
  the allocation's available cores and the site's reserved-core guidance.

For device discovery and sampled GPU activity, consult the site's
[`xpu-smi` guide](https://docs.alcf.anl.gov/aurora/performance-tools/xpu-smi/)
and check tool availability in the compute-node environment.

## ChemGraph Parsl workers

`get_aurora_config` requires `PBS_NODEFILE` and uses `LocalProvider` with
`MpiExecLauncher` inside an existing allocation. It does not submit PBS jobs.
It configures twelve accelerator slots per node and defaults to nine workers;
adjust concurrency with `max_workers_per_node` or
`CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE` for the calculation's memory needs.
Verify the installed Parsl version's device assignment with the calculator
and GPU hierarchy before a large run.

Worker setup selects `CHEMGRAPH_WORKER_INIT`, then a detected Python environment,
then the `module load frameworks` fallback. If the detected environment needs
base modules, include them in `CHEMGRAPH_WORKER_INIT`. An explicit `worker_init`
replaces the full setup snippet, including directory and environment setup.
Check that inputs, environments and output directories are visible to every
worker; the agent shell and remote workers can have different filesystems.
