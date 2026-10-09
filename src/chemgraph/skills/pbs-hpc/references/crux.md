# Crux

Crux is a CPU system. Use the [machine guide](https://docs.alcf.anl.gov/crux/)
and [job guide](https://docs.alcf.anl.gov/crux/queueing-and-running-jobs/running-jobs/)
for node layout and current queue policy. Check `qstat -Qf QUEUE` before
submission; preserve supplied resource choices for prepare-only work.

## Prepare and launch calculations

- Use `select=N:system=crux`, the supplied project/queue/walltime and required
  filesystems, for example `home:eagle`. Stage inputs, model weights and the
  Python environment on storage visible to compute nodes; keep persistent
  results on shared project storage.
- Select a CPU-supported calculator and environment. Confirm ChemGraph and its
  calculator dependencies are available after activation. Set `TMPDIR=/tmp`
  and change to the run directory before launching.
- Use the parent skill's [PBS template](../assets/job.pbs.template) for direct
  PBS or the `iri-hpc` staging workflow for IRI. Both use the
  [allocation helper](../assets/pbs-launch.sh).
  Obtain the application command and results contract from
  [ASE calculations](../../chemgraph/references/ase-calculations.md).
- Set `OMP_NUM_THREADS` explicitly and configure any BLAS thread pools used by
  the calculator. Keep concurrent workers times threads per worker within the
  allocated CPU budget, and account for each worker's memory requirements.
  Inspect `lscpu` or `numactl --hardware` on a compute node before tuning affinity.
- Use the site's `mpiexec` for MPI applications, with explicit ranks per node,
  thread count and CPU binding. Independent calculations need separate inputs
  and output directories; launching identical ASE runners with shared outputs
  is not a parallelization strategy. Run calculations and substantial environment
  checks within an allocation.

## ChemGraph Parsl workers

`get_crux_config` requires `PBS_NODEFILE` and uses `LocalProvider` with
`MpiExecLauncher` inside an existing allocation. It does not submit PBS jobs.
The default `max_workers_per_node=16` controls concurrent workers; it does not
set the calculator's thread count. Tune both for the CPU and memory budget.

Worker setup selects `CHEMGRAPH_WORKER_INIT`, then a detected Python environment,
then the `module load conda; conda activate base` fallback. The fallback does
not guarantee that ChemGraph or the calculator is installed. Supply the intended
environment explicitly when needed. An explicit `worker_init` replaces the full
setup snippet, including directory and environment setup. Confirm that every
worker can read inputs and write its own results.
