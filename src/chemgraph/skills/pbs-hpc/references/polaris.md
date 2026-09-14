# ALCF Polaris

Polaris is a 560-node HPE Apollo 6500 Gen 10+ system. Each compute node: one 2.8 GHz AMD
EPYC Milan 7543P (32 cores / 64 threads), 512 GiB DDR4, **4 NVIDIA A100 GPUs** (40 GiB
HBM2 each, NVLink), 2x 1.6 TB local SSD (RAID0), 2x Slingshot 11 NICs. Scheduler is
**PBS Pro**.

## Documentation source

Use the [official Polaris guide](https://docs.alcf.anl.gov/polaris/) for current site
information. When a deployment provides a local checkout of `argonne-lcf/user-guides`,
read `<user-guides-checkout>/docs/polaris` and check its freshness against upstream.
Obtain the checkout path from the user or deployment configuration; it is not a fixed
path shared by all Polaris users.

When a question is not fully answered below, search the relevant local guide files or
consult the official site. Key files:

- `getting-started.md` — login, modules, proxy
- `index.md` — machine/hardware overview, device affinity table, login nodes
- `running-jobs/index.md` — queues, interactive jobs, mpiexec, node/rack topology
- `running-jobs/using-gpus.md` — GPU affinity script, MPS, MIG
- `compiling-and-linking/`, `programming-models/`, `data-science/`,
  `applications-and-libraries/`, `workflows/`, `containers/`, `known-issues.md`,
  `system-updates.md`

## Logging in

```bash
ssh your-username@polaris.alcf.anl.gov   # password = CRYPTOCard / MobilePASS+ token
```

Four user login nodes (`polaris-login-01..04`), **shared by all users**. Build code and
submit jobs here, but **do not run heavy compute or I/O on login nodes**. Login nodes
have **no GPUs** — if a build needs a physical GPU, build on a compute node via an
interactive job.

**Be careful on login nodes** — they are a shared resource:

- Do not walk/stat/scan large directory trees (`find`, `ls -R`, `du`, `grep -r`,
  globbing over huge dirs). Metadata-heavy operations on Lustre hammer the metadata
  targets and slow the system for everyone. Do bulk file operations from a compute node
  instead.
- Do not spawn many processes — keep build parallelism reasonable (e.g. `make -j8`, not
  `-j$(nproc)` which is 256 threads), and avoid launching swarms of
  scripts/subprocesses.
- Do not run compute- or I/O-intensive pre/post-processing on logins. Get an interactive
  compute node for that.
- Keep intensive I/O on the parallel project filesystems (`/eagle`, `/grand`), not
  `/home`.

## Software / modules

```bash
module use /soft/modulefiles   # expose ALCF-installed software in /soft
module avail                   # list available modules
```

A Spack-based PE is also available (see
`applications-and-libraries/libraries/spack-pe.md`). Consult [getting
started](https://docs.alcf.anl.gov/polaris/getting-started/) for current login and
software setup.

## Internet proxy (login and compute nodes)

Compute nodes reach the internet only via a proxy. Set before downloading
(pip/git/conda/wget):

```bash
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

Use the full current `no_proxy` exclusions in the [getting-started
guide](https://docs.alcf.anl.gov/polaris/getting-started/) so internal connections do
not go through the external proxy.

## Queues

Submit with `qsub -q QUEUE -A PROJECT`. The [job
guide](https://docs.alcf.anl.gov/polaris/running-jobs/) lists these six queues. Limits
were checked on 2026-09-13; confirm current settings before submission. Minimum walltime
is five minutes.

| Queue         | Node min | Node max | Walltime max | Notes |
|---------------|----------|----------|--------------|-------|
| `debug`       | 1        | 2        | 1 hr         | 8 dedicated nodes (up to 24 if free); max 24 nodes in-queue |
| `debug-scaling` | 1      | 10       | 1 hr         | max 1 job running/queued per user |
| `prod`        | 10       | 496      | 24 hr        | **routing queue** → small/medium/large (submit here, not to exec queues) |
| `preemptable` | 1        | 10       | 72 hr        | **can be killed anytime**; add `#PBS -r y` to rerun; max 20 jobs/project |
| `demand`      | 1        | 56       | 1 hr         | by request only; preempts `preemptable` |
| `capacity`    | 1        | 4        | 168 hr       | max 32 nodes across all jobs; at most 1 running and 2 queued-or-running jobs total per user |

`prod` routes by node count: small (10–24, 3 hr), medium (25–99, 6 hr), large (100–496,
24 hr). Additional backfill tiers are available. For near-full-system jobs, check
current node availability instead of relying on a fixed spare-node recommendation.
Inspect a queue: `qstat -Qf QUEUE`.

Use `#PBS -r y` for a preemptable job only when restarting the job from the beginning or
its checkpoints is safe for the application and output files.

## Interactive job

```bash
PROJECT="your-project"
qsub -I -l select=1:system=polaris -l filesystems=home:eagle \
    -l walltime=1:00:00 -q debug -A "$PROJECT"
```

Declare the filesystems the job actually needs, for example `-l filesystems=home:eagle`;
a missing or incorrect filesystem list can leave jobs held. To `ssh`/`scp` to assigned
compute nodes, `$HOME` and `$HOME/.ssh` must both be mode `700`.

## Batch script + MPI launch

Launch with `mpiexec` (not `srun`/`aprun`). Useful flags: `-n` total ranks, `--ppn`
ranks/node, `--depth` CPUs/rank, `--cpu-bind`, `--env`, `--hostfile` (defaults to
`$PBS_NODEFILE`).

Replace `PROJECT_NAME` and load the application environment before submission. This
example uses four nodes, eight ranks per node, and eight OpenMP threads per rank; select
the layout for the application.

```bash
#!/bin/bash -l
#PBS -N myjob
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A PROJECT_NAME
#PBS -j oe

# Load the application modules/environment here.
set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS submission directory is required}"
NNODES=$(wc -l < "${PBS_NODEFILE:?An active PBS allocation is required}")
NRANKS_PER_NODE=8
NDEPTH=8            # 64 hardware threads / NRANKS_PER_NODE
NTHREADS=8
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

mpiexec -n "$NTOTRANKS" --ppn "$NRANKS_PER_NODE" --depth "$NDEPTH" \
    --cpu-bind depth --env OMP_NUM_THREADS="$NTHREADS" ./my_app
```

Validate the completed script with `bash -n` and follow the parent skill's submission
and monitoring workflow.

## GPU usage & affinity

Follow the [Polaris GPU
guide](https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/). Validate the
application's accelerator support inside the compute environment; not every calculator
supports GPUs.

- `MPICH_GPU_SUPPORT_ENABLED=1` for GPU-aware MPI; also load
  `craype-accel-nvidia80` with `module load` at compile and runtime (else `GTL library is not linked`
  errors).
- Restrict GPUs with `CUDA_VISIBLE_DEVICES` (e.g. `0,1`).
- Cray MPI does **not** bind ranks to GPUs. Use a wrapper placed before the executable.
  Note the **reverse** GPU order (topology): closest GPU to a NUMA node is assigned
  round-robin.

```bash
#!/bin/bash -l
# Save as set_affinity_gpu_polaris.sh in the execution workspace.
: "${PMI_LOCAL_RANK:?Run this wrapper under mpiexec}"
num_gpus=4
gpu=$(( (num_gpus - 1) - (PMI_LOCAL_RANK % num_gpus) ))
export CUDA_VISIBLE_DEVICES="$gpu"
exec "$@"
```

```bash
mpiexec -n "$NTOTRANKS" --ppn "$NRANKS_PER_NODE" --depth "$NDEPTH" \
    --cpu-bind depth bash ./set_affinity_gpu_polaris.sh ./my_app
```

This wrapper assumes access to all four GPUs on each node. Four ranks per node gives one
rank per GPU; more ranks share GPUs. Adapt the layout for applications using multiple
GPUs per rank. Preserve launcher-managed assignments, restricted device sets, and MIG
configurations instead of overwriting them with physical GPU indices.

Device affinity (from `index.md`): GPU0↔CPU 24-31,56-63 (NUMA3); GPU1↔16-23,48-55
(NUMA2); GPU2↔8-15,40-47 (NUMA1); GPU3↔0-7,32-39 (NUMA0).

- **MPS** (multiple processes per GPU): start `nvidia-cuda-mps-control -d` per node (one
  rank/node via `mpiexec -n "$NNODES" --ppn 1`); give each node its own pipe/log
  directories. Clients and the control daemon must use matching
  `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` values. Follow the complete
  startup and shutdown sequence in `using-gpus.md`.
- **MIG** (partition a GPU): `qsub -l mig_config=/path/mig_config.json ...`; only in
  debug/debug-scaling/preemptable queues; validate with
  `/soft/pbs/mig_conf_validate.sh -c config.json`. Verify devices in an interactive allocation: invalid configurations
  or unsupported queues can silently ignore the request.

## Filesystems

Consult the [storage
guide](https://docs.alcf.anl.gov/data-management/filesystem-and-storage/) for current
mounts, quotas, and backup policies. The shared filesystems below use **Lustre**;
node-local SSD storage uses XFS. Declare the ones a job needs:
`-l filesystems=home:eagle` (jobs may be held if this is missing or wrong).

| Name        | Path                              | Type | Backed up | Quota / notes |
|-------------|-----------------------------------|------|-----------|----------------|
| agile-home  | `/home` (`/lus/agile/home`)       | Lustre | **Yes** (to tape) | 50 GB default per user. Small files + binaries only. |
| eagle       | `/eagle` (`/lus/eagle/projects`)  | Lustre | No | Per-**project** directory quota (not per-user). Large files, intensive job I/O, Globus sharing. |
| grand       | `/grand` (`/lus/grand/projects`)  | Lustre | No | Per-project where available; confirm access and compute-node mounts before use. |
| Node SSD    | `/local/scratch` (compute only)   | xfs  | No | Two 1.6 TB drives per node; check usable free space. Non-parallel. **Wiped between jobs.** Best for heavy per-node scratch I/O. |

The [Globus transfer
guide](https://docs.alcf.anl.gov/data-management/data-transfer/using-globus/) also
documents Grand access. Do not assume every project has storage on each filesystem.

Guidance:

- **`/home` is for small files and binaries.** Its performance is fine for that, but
  doing intensive I/O from compute nodes against `/home` is discouraged — route heavy
  I/O to `/eagle` or `/grand`, which are fast parallel systems with far more space.
- Project directories have a **directory quota** (all files under the project dir count,
  regardless of owner), not a per-user quota. See
  `data-management/filesystem-and-storage/disk-quota.md`.
- Only `agile-home` is backed up. **The project data filesystems are NOT backed up** —
  archive critical data to tape (HPSS) or elsewhere.
- `/local/scratch` and `/dev/shm` are good per-node scratch; copy required results out
  before the job ends. Include node names in staged paths when combining output from
  multiple nodes.
- **Lustre is sensitive to metadata load** — avoid creating huge numbers of tiny files
  or scanning enormous directory trees, especially from login nodes (see the login-node
  cautions above). For many-file workloads, use the node-local SSD or tar/bundle files.

## Common commands

- `qstat -u "$USER"` — your jobs; `qstat -f JOB_ID` — job details; `qstat -Qf QUEUE` —
  queue details
- `qdel JOB_ID` — cancel the requested job; `nvidia-smi` / `nvidia-smi topo -m` — GPU
  status/topology (on a compute node)
- `sbank` — allocation usage

## ChemGraph execution

For the complete login-node ASE MCP + PBS-managed Parsl example, and a direct
batch alternative, read [ASE calculations](polaris-ase.md). The explicit
`allocation_mode="pbs"` uses `PBSProProvider` to acquire nodes from a login node.
The existing configuration described below remains the default.

ChemGraph's `get_polaris_config` uses `PBS_NODEFILE` to size an existing allocation and
launches workers using `LocalProvider` and `MpiExecLauncher`. It configures four
accelerators and the CPU affinity groups above. Its one-node fallback outside PBS is for
local/testing contexts and does not submit a scheduler job. Preserve the
application-specific MPI/GPU layout and configure worker setup through
`CHEMGRAPH_WORKER_INIT` or the configured Python environment.

Skill files, the `execute` shell, an HPC MCP server, and compute workers can have
different filesystems. Copy templates and helper scripts into the actual execution
workspace and check worker visibility.

Inspect stdout/stderr, application results, and available scheduler history to establish
success; a job disappearing from the active queue does not prove completion.
