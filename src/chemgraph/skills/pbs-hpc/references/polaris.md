# ALCF Polaris

Polaris: 560 nodes, PBS Pro. Each compute node has one 32-core AMD EPYC Milan CPU
(64 threads), 512 GiB DDR4, **4 NVIDIA A100 GPUs** (40 GiB each), and local SSD. Use the
[official Polaris guide](https://docs.alcf.anl.gov/polaris/) for anything not covered
here; a deployment may also provide a local `argonne-lcf/user-guides` checkout
(`docs/polaris`), whose path comes from the user or configuration.

## Logging in

`ssh USER@polaris.alcf.anl.gov` (MobilePASS+ token). The four login nodes are shared
and have no GPUs: build and submit there, but run calculations, downloads of large
models, and heavy pre/post-processing in a job.

- Do not scan large directory trees (`find`, `ls -R`, `du`, `grep -r`) on Lustre.
- Keep build parallelism modest (`make -j8`) and avoid process swarms.
- Keep intensive I/O on `/eagle` or `/grand`, not `/home`.

## Software / modules

`module use /soft/modulefiles` exposes ALCF software; see [getting
started](https://docs.alcf.anl.gov/polaris/getting-started/) for current setup.

### ChemGraph environment

ChemGraph jobs use the shared environment
`/grand/catalyst/world-shared/chemgraph/alcf_chemgraph_env` unless the user names
another; do not ask for one. Activate it with `source <env>/bin/activate` before
`set -u`, run `<env>/bin/python`, and declare `grand` with the other filesystems
(`home:eagle:grand`). If it is unavailable on compute nodes, report that rather than
installing or switching environments. The
[Polaris PBS template](../assets/polaris-chemgraph.pbs.template) and IRI's
`iri_prepare.py` already apply these settings.

## Internet proxy (login and compute nodes)

Compute nodes reach the internet only through a proxy, for example to download named
model weights. The templates set:

```bash
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
```

Use the full `no_proxy` list from the getting-started guide when a job also needs
internal ALCF services.

## Queues

Submit with `qsub -q QUEUE -A PROJECT`. Limits from the [job
guide](https://docs.alcf.anl.gov/polaris/running-jobs/), checked 2026-09-13; confirm with
`qstat -Qf QUEUE`. Minimum walltime is five minutes.

| Queue         | Nodes  | Walltime max | Notes |
|---------------|--------|--------------|-------|
| `debug`       | 1–2    | 1 h          | default for single ChemGraph calculations |
| `debug-scaling` | 1–10 | 1 h          | max 1 job running/queued per user |
| `prod`        | 10–496 | 24 h         | routing queue (small/medium/large by node count) |
| `preemptable` | 1–10   | 72 h         | can be killed anytime; `#PBS -r y` only if restart is safe |
| `demand`      | 1–56   | 1 h          | by request only |
| `capacity`    | 1–4    | 168 h        | at most 1 running and 2 queued-or-running jobs per user |

## Interactive job

```bash
qsub -I -l select=1:system=polaris -l filesystems=home:eagle:grand \
    -l walltime=1:00:00 -q debug -A "$PROJECT"
```

A missing or wrong `filesystems` list can leave jobs held. To `ssh` to assigned compute
nodes, `$HOME` and `$HOME/.ssh` must be mode `700`.

## MPI and GPUs

A single ChemGraph calculation runs one process on GPU 0 (`CUDA_VISIBLE_DEVICES=0`,
`OMP_NUM_THREADS=8`); no `mpiexec` is needed. For multi-rank or multi-GPU applications,
launch with `mpiexec` (`-n`, `--ppn`, `--depth`, `--cpu-bind`) and follow the [GPU
guide](https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/) for the affinity
wrapper, MPS, and MIG. Cray MPI does not bind ranks to GPUs by itself.

## Filesystems

Declare every filesystem the job touches, for example `-l filesystems=home:eagle:grand`.
See the [storage guide](https://docs.alcf.anl.gov/data-management/filesystem-and-storage/)
for quotas and policies.

| Name       | Path                             | Backed up | Use |
|------------|----------------------------------|-----------|-----|
| home       | `/home` (`/lus/agile/home`)      | Yes       | small files and binaries (50 GB) |
| eagle      | `/eagle` (`/lus/eagle/projects`) | No        | run directories, job I/O, Globus sharing; per-project quota |
| grand      | `/grand` (`/lus/grand/projects`) | No        | per-project storage; holds the shared ChemGraph environment |
| Node SSD   | `/local/scratch` (compute only)  | No        | per-node scratch, wiped between jobs |

- Project filesystems are not backed up; archive critical data elsewhere.
- Lustre is sensitive to metadata load; avoid many tiny files.
- Do not assume every project has storage on each filesystem.

## Common commands

- `qstat -u "$USER"`, `qstat -f JOB_ID`, `qstat -xf JOB_ID` (finished jobs),
  `qstat -Qf QUEUE`
- `qdel JOB_ID` — cancel only the requested job
- `nvidia-smi` on a compute node; `sbank` for allocation usage

## ChemGraph execution

ChemGraph's `get_polaris_config` sizes an existing allocation from `PBS_NODEFILE` and
starts workers with `LocalProvider` and `MpiExecLauncher`; its one-node fallback outside
PBS is for testing and submits nothing. Configure worker setup through
`CHEMGRAPH_WORKER_INIT` or the configured Python environment.
