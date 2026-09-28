# Polaris: routine batch jobs

Use [the official job guide](https://docs.alcf.anl.gov/polaris/running-jobs/)
for current queue limits before submission and
[getting started](https://docs.alcf.anl.gov/polaris/getting-started/) for environment
setup. Preserve supplied project/queue/resource values for prepare-only work;
do not invent an allocation or infer a GPU requirement from the site name.

- Use `select=N:system=polaris`, the user's queue/project/walltime and the
  filesystems the job needs, for example `home:eagle`. Missing filesystem
  declarations can leave jobs held. Inputs and environment paths must be visible
  on compute nodes. Resolve host paths rather than using virtual `/workspace` paths.
- Source the requested environment; temporarily disable nounset if activation
  requires it, then restore it. Change to the run directory and set `TMPDIR=/tmp`
  before launching the application. Retain site proxy
  settings; download/stage model weights before the calculation.
- **One CPU process:** supply the selected executable to the allocation helper with the
  application's requested thread count. CPU work does not need CUDA setup or an
  MPI launcher.
- **One GPU process:** use the requested GPU-capable calculator/environment;
  the ASE example can use `CUDA_VISIBLE_DEVICES=0` and `OMP_NUM_THREADS=8`.
- **MPI, multiple GPUs or environment problems:** read
  [advanced operations](polaris-advanced.md), including launch/affinity, modules,
  proxies, MPS/MIG, queues and storage. MPI launch uses `mpiexec`, not `srun`/`aprun`.

Use [ASE calculations](../../chemgraph/references/ase-calculations.md) for calculation
inputs and the runner. The [allocation helper](../assets/pbs-launch.sh) guards
execution for both direct PBS and IRI jobs. Login nodes are shared: do not run calculations,
model initialization or heavy filesystem scans there. Use an allocation for heavy
work; keep intensive I/O on project storage. Prepare-only validation does not
prove environment readiness on compute nodes.
