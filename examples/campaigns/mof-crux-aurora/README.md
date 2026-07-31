# mof-crux-aurora

Cross-HPC agentic materials-discovery demo. Crux plans and preps a MOF,
hands off to Aurora for GCMC simulation, gets the uptake back. Real
science tools, no mocks.

Pipeline (`~15 min` end-to-end, UMA optimization dominates):

- **Crux `planner_executor`** — `mofforge_build` (pormake topology
  assembly) → `run_ase` (UMA fairchem geometry optimization) →
  `pacmof2_assign_charges` (PACMOF2 partial charges) → Globus transfer
  to Aurora → message aurora_sim → wait for reply → finish.
- **Aurora `aurora_sim`** — receive message → `run_graspa_ensemble`
  (SYCL gRASPA GCMC uptake) → Globus transfer result back to Crux →
  reply → finish.

## Prereqs

Do these once per operator per HPC.

### Both HPCs

- `~/.globus/chemgraph_transfer_tokens.json` present on both. Mint on
  laptop with the ChemGraph `GlobusTransferManager` (any endpoint pair
  triggers the browser flow), then `scp` to both HPCs. Consent scopes:
  `data_access` on both `alcf#dtn_eagle` and `alcf#dtn_flare`.
  Refresh every ~48 h — the refresh flow runs automatically on first
  MCP call but only if the file has a valid `refresh_token`.
- Inbox + output dirs:

  ```bash
  ssh <user>@crux.alcf.anl.gov  "mkdir -p /eagle/ChemGraph/<user>/{inbox-crux,inbox-aurora,mof-out/{build,charges}}"
  ssh <user>@aurora.alcf.anl.gov "mkdir -p /flare/ChemGraph/<user>/{inbox-aurora,mof-out/graspa}"
  ```

### Crux venv (`/eagle/ChemGraph/<user>/venvs/academy-dev`, Python 3.11)

- ChemGraph editable-install of this branch:

  ```bash
  git clone https://github.com/argonne-lcf/ChemGraph.git /eagle/ChemGraph/<user>/ChemGraph
  cd /eagle/ChemGraph/<user>/ChemGraph && git checkout dev
  source /eagle/ChemGraph/<user>/venvs/academy-dev/bin/activate
  pip install -e . --no-deps
  ```

- Swarm launcher package (from your laptop `academy/` checkout, or
  clone the equivalent source; installed editable in the same venv):

  ```bash
  # From laptop, rsync ~/projects/chemgraph-academy/academy/ to
  # /eagle/ChemGraph/<user>/swarm/ once, then:
  pip install -e /eagle/ChemGraph/<user>/swarm --no-deps
  ```

- MLIP / real-tool deps (all `--no-deps` to avoid clobbering the
  numpy/scipy pins with pytorch's CUDA cascade):

  ```bash
  pip install --no-deps fairchem-core
  pip install --no-deps git+https://github.com/snurr-group/pacmof2.git
  pip install --no-deps torchtnt pyre_extensions typing_inspect \
    submitit e3nn mypy_extensions ml_dtypes opt_einsum \
    ray wandb huggingface_hub hf-xet
  pip install --no-deps -e /eagle/ChemGraph/<user>/mofforge   # assemble backend
  pip install --no-deps pormake jax jaxlib                     # topology BBs
  ```

  Torch is CPU-only for Crux:

  ```bash
  pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch
  ```

  Smoke-test all imports resolve:

  ```bash
  OPENBLAS_NUM_THREADS=4 python -c "
  import chemgraph
  from chemgraph.mcp import pacmof2_mcp_hpc, fairchem_mcp_hpc, graspa_mcp_hpc  # noqa
  import mofforge, pacmof2, fairchem.core, pormake  # noqa
  print('crux stack ok')"
  ```

### Aurora venv (`/flare/ChemGraph/<user>/venvs/academy-dev`, Python 3.12, `--system-site-packages`)

Aurora's `frameworks` module ships XPU-native torch — inherit it via
`--system-site-packages` instead of installing your own torch:

```bash
module load frameworks
python -m venv --system-site-packages /flare/ChemGraph/<user>/venvs/academy-dev
source /flare/ChemGraph/<user>/venvs/academy-dev/bin/activate
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

Then the same install shape as Crux (skip torch since inherited):

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git /flare/ChemGraph/<user>/ChemGraph
cd /flare/ChemGraph/<user>/ChemGraph && git checkout dev
pip install -e . --no-deps
pip install -e /flare/ChemGraph/<user>/swarm --no-deps

pip install --no-deps fairchem-core
pip install --no-deps git+https://github.com/snurr-group/pacmof2.git
pip install --no-deps torchtnt pyre_extensions typing_inspect \
  submitit e3nn mypy_extensions ml_dtypes opt_einsum \
  ray wandb huggingface_hub hf-xet
[ -d /flare/ChemGraph/<user>/mofforge ] && pip install --no-deps -e /flare/ChemGraph/<user>/mofforge
```

### Aurora-only: gRASPA-sycl binary

```bash
cd /flare/ChemGraph/<user>
[ -d gRASPA/graspa-sycl ] || git clone https://github.com/snurr-group/gRASPA-fast.git gRASPA/graspa-sycl
cd gRASPA/graspa-sycl && mkdir -p build && cd build
cmake -DUSE_SYCL=1 .. && make -j 8
ls -la ../bin/sycl.out    # produced binary
```

Point the MCP wrapper at it via profile env:

```jsonc
// examples/profiles/aurora.template.json (this repo)
"env": {
  ...,
  "CHEMGRAPH_GRASPA_BIN": "/flare/${ALCF_PROJECT}/${ALCF_USER}/gRASPA/graspa-sycl/bin/sycl.out"
}
```

### Argo-shim (laptop)

`swarm dashboard` needs a local `argo-shim` on your laptop at
`http://127.0.0.1:18085`. Start it once before launching:

```bash
CELS_USERNAME=<your.cels.login> argo-shim --no-auth --no-update-settings --port 18085 --tunnel
```

See `examples/connecting_to_argo/README.md` for the Duo/tunnel setup
and the compute-node-relay variant.

## Launch

```bash
export ALCF_PROJECT=ChemGraph ALCF_USER=<user> \
       ALCF_SSH_USER=<sshuser> ARGO_USER=<argo>
RUN_ID="mof-crux-aurora-$(date +%Y%m%d-%H%M)"
swarm dashboard -- "$RUN_ID" \
  --system crux,aurora --enable-launch-buttons \
  --bundle-root "/eagle/${ALCF_PROJECT}/${ALCF_USER}/ChemGraph" \
  --project "${ALCF_PROJECT}"
```

Open the dashboard URL (default `http://127.0.0.1:8765`):

1. Canvas tab → campaign dropdown → `mof-crux-aurora`.
2. Verify agent-to-site placement: `planner_executor → crux`,
   `aurora_sim → aurora`. Drag if wrong.
3. Click **Launch crux** and **Launch aurora**. Both spawn PBS jobs
   via ssh to the login node; the dashboard polls until both go `R`
   and both agents write `agent_status/<name>.json`.
4. Once both are green, open the **Inject** panel (right side), pick
   `planner_executor` as recipient, paste the kickoff JSON (replace
   `<user>` and the pormake bbs path — mine is under the `academy-dev`
   venv's `pormake/database/bbs/`):

   ```json
   {
     "spec": {
       "topology": "pcu",
       "backend": "pormake",
       "node_files": ["/eagle/ChemGraph/<user>/venvs/academy-dev/lib/python3.11/site-packages/pormake/database/bbs/N59.xyz"],
       "edge_files": ["/eagle/ChemGraph/<user>/venvs/academy-dev/lib/python3.11/site-packages/pormake/database/bbs/E32.xyz"],
       "output_dir": "/eagle/ChemGraph/<user>/mof-out/build"
     },
     "opt_output_json": "/eagle/ChemGraph/<user>/mof-out/opt.json",
     "charges_output_dir": "/eagle/ChemGraph/<user>/mof-out/charges",
     "adsorbate": "H2O",
     "temperature_k": 298.15,
     "pressure_pa": 101325.0
   }
   ```

## What you should see

Observability tab, roughly in order:

1. `mofforge_build` on Crux → CIF at `mof-out/build/pcu_N59_E32.cif`.
2. `run_ase` (UMA opt on CPU, ~15 min).
3. `pacmof2_assign_charges` → charged CIF at `mof-out/charges/`.
4. `transfer_file` → Globus from
   `alcf#dtn_eagle:/ChemGraph/<user>/mof-out/charges/<name>.cif` to
   `alcf#dtn_flare:/ChemGraph/<user>/inbox-aurora/<name>.cif`.
5. `send_message(recipient='aurora_sim', ...)`.
6. **aurora_sim** wakes: `run_graspa_ensemble` (SYCL GCMC on XPU),
   `transfer_file` (result JSON back to Crux inbox), `send_message`
   reply, `finish_turn`.
7. **planner_executor** receives reply with uptake numbers, then
   `finish_turn`.

Final artifacts:

- Crux: `mof-out/{build,charges}/*.cif`, `inbox-crux/*.raspa.json`
- Aurora: `inbox-aurora/*.cif`, `mof-out/graspa/*.raspa.json`

## Endpoint UUIDs

Hardcoded in the missions:

- `alcf#dtn_eagle` (Crux Eagle DTN, DTN-root `/ChemGraph/`):
  `05d2c76a-e867-4f67-aa57-76edeb0beda0`
- `alcf#dtn_flare` (Aurora Flare DTN, DTN-root `/ChemGraph/`):
  `f39a7a0f-5bfc-46ce-9615-ba9f8592814f`

Filesystem paths for local tools use the site's mount (`/eagle/...` on
Crux, `/flare/...` on Aurora); Globus source/dest paths use the
DTN-relative form (`/ChemGraph/...`).

## Design notes worth knowing

- **`mcp_servers` is a library, not a per-site inventory.** Each
  agent's `mcp_servers: [...]` whitelists which library entries its
  daemon spawns. See `daemon.py:72-77` (`[spec for spec in
  campaign.mcp_servers if spec.name in agent_spec.mcp_servers]`).
  Nothing idles — servers not in the running agent's whitelist are
  never started on that site.
- **`$VIRTUAL_ENV/bin/python` in commands** is expanded at spawn time
  from the daemon's own `sys.executable`, so one shared library entry
  lands at `/eagle/.../academy-dev/bin/python` on Crux and
  `/flare/.../academy-dev/bin/python` on Aurora without editing the
  JSON per site.
- **Site → MCP mapping is implicit** through which agent runs where,
  driven by the launcher's `--agents` slice (set by dashboard
  swimlane assignment or the CLI's `--agents` flag).

## Known limits

- The multi_agent planner sometimes hallucinates tools like `gather`
  or `submit_result`; the runtime returns errors and it moves on.
  Noise, not a blocker.
- After the sim's reply the planner doesn't always self-call
  `finish_turn`; PBS walltime terminates cleanly. Cosmetic.
