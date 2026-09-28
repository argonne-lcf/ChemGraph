# Polaris

Use the [official job guide](https://docs.alcf.anl.gov/polaris/running-jobs/)
and `qstat -Qf QUEUE` for current queue limits. Preserve supplied project, queue,
walltime and resource choices for prepare-only work. Select CPU or GPU execution
from the calculation's requirements.

## Prepare the calculation environment

- Use `select=N:system=polaris` and declare required filesystems, for example
  `home:eagle`. Stage inputs, model weights and outputs in the actual shared
  project directory; confirm that compute nodes can read the Python environment.
  Virtual `/workspace` paths are not automatically remote filesystem paths.
- Load the requested modules and activate the environment before enabling
  nounset. Check `module list` and the selected Python executable. Set
  `TMPDIR=/tmp` and change to the run directory before launching.
- Retain the site's proxy settings when network access is needed, including
  internal-host exclusions from the
  [environment guide](https://docs.alcf.anl.gov/polaris/getting-started/#proxy).
  Stage model downloads before starting the calculation.
- Use allocated compute nodes for calculator initialization, GPU checks and
  substantial computation. Keep intensive I/O on project storage. Copy required
  outputs from node-local scratch to shared storage before the job ends.

## Launch and GPU checks

Use the parent skill's [PBS template](../assets/job.pbs.template) for direct PBS
or the `iri-hpc` staging workflow for IRI. Both use the
[allocation helper](../assets/pbs-launch.sh).
Obtain inputs, the runner and result expectations from
[ASE calculations](../../chemgraph/references/ase-calculations.md).

- **One CPU process:** launch the application with its requested thread count.
- **One GPU process:** confirm the calculator and Python environment support
  CUDA. Polaris has four NVIDIA A100 GPUs per compute node. An unrestricted
  single-process job can select one with `CUDA_VISIBLE_DEVICES=0`; preserve
  device assignments already supplied by a worker or launcher.
- **MPI application:** use the site's `mpiexec`; set total ranks, ranks per
  node, CPU binding and threads to match the application. Multiple MPI ranks
  do not automatically distribute an ASE input: independent calculations need
  separate work directories and inputs, or a configured ensemble backend.
- **Multiple GPUs:** confirm each process's selected device. Use the
  [site GPU-binding examples](https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/)
  when the application needs a rank-to-device wrapper. Match CPU affinity to
  GPU topology; inspect `nvidia-smi topo -m` within the allocation.
- **GPU-aware MPI:** only applications passing GPU buffers to MPI need
  `MPICH_GPU_SUPPORT_ENABLED=1`. Their build and runtime must include the
  `craype-accel-nvidia80` module for the GPU transport library. This is not a
  prerequisite for a single-process CUDA calculator.

Use the site GPU guide for MPS or MIG only when the requested workload needs
GPU sharing. Check the resulting visible devices before applying a binding
wrapper; a fixed physical-GPU mapping may be wrong for a restricted device set.

## ChemGraph Parsl workers

`get_polaris_config` uses `PBS_NODEFILE`, `LocalProvider` and `MpiExecLauncher`
inside an existing allocation. It configures four accelerators per node with
CPU affinity groups matched to GPU topology. Its one-node fallback without
PBS is for local/testing contexts and does not obtain an allocation.

Configure worker setup through `worker_init` or `CHEMGRAPH_WORKER_INIT`, or
use the detected Python environment. The fallback loads no application modules,
so ensure ChemGraph and the calculator are available on every worker. An explicit
`worker_init` replaces the full setup snippet: include the working directory and
temporary-directory setup as well as environment activation.
