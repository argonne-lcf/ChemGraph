# mof-crux-aurora

Cross-HPC agentic materials-discovery demo: Crux plans + preps a MOF, hands
off to Aurora for GCMC simulation, gets the uptake back.

Two campaigns ship in the same shape:

- `mof-crux-aurora` — real science tools (mofforge, UMA opt, PACMOF2, gRASPA).
  Full pipeline takes ~15 min (UMA opt dominates).
- `mof-crux-aurora-mock` — schema-matched mocks under
  `chemgraph.academy.tools.mock_mcp`. Full pipeline takes <1 min. Use this
  for iteration and reviewer reproducibility; the LLM cannot tell the
  tools are mock (same names, same schemas, deterministic plausible
  outputs, real files written).

## What you need

- **Two HPC allocations**: Crux (`ChemGraph` project on Eagle) + Aurora
  (`ChemGraph` project on Flare).
- **Two venvs** installed with this branch:
  - Crux: `/eagle/ChemGraph/<user>/venvs/academy-swarm-crux` (Python 3.11)
  - Aurora: `/flare/ChemGraph/<user>/venvs/academy-swarm` (Python 3.12,
    frameworks module loaded). Aurora also needs `pip install -e
    /flare/ChemGraph/<user>/ChemGraph` for `chemgraphagent`.
- **`~/.globus/chemgraph_transfer_tokens.json`** on your laptop AND on
  Aurora (Crux/Aurora don't share `$HOME`). Mint the token file once by
  running `chemgraph.execution.globus_transfer.GlobusTransferManager`
  from any interactive login shell, then `scp` it to Aurora.
  Consent-scopes: `data_access` on both `alcf#dtn_eagle` and
  `alcf#dtn_flare`.
- **argo-shim** running on your laptop at `http://127.0.0.1:18085`.
- **Inbox + output dirs** on both filesystems:

  ```bash
  ssh <user>@crux.alcf.anl.gov  "mkdir -p /lus/eagle/projects/ChemGraph/<user>/{inbox-crux,inbox-aurora,mof-out/{build,charges}}"
  ssh <user>@aurora.alcf.anl.gov "mkdir -p /flare/ChemGraph/<user>/{inbox-aurora,mof-out/graspa}"
  ```

## Launch

```bash
export ALCF_PROJECT=ChemGraph ALCF_USER=<user> ALCF_SSH_USER=<sshuser> ARGO_USER=<argo>
RUN_ID="mof-crux-aurora-mock-$(date +%Y%m%d-%H%M)"
swarm dashboard -- "$RUN_ID" \
  --system crux,aurora --enable-launch-buttons \
  --bundle-root "/eagle/${ALCF_PROJECT}/${ALCF_USER}/ChemGraph" \
  --project "${ALCF_PROJECT}"
```

Open the dashboard URL (default `http://127.0.0.1:8765`):

1. Canvas tab → campaign dropdown → `mof-crux-aurora-mock` (or `mof-crux-aurora` for real).
2. Verify agent-to-site assignment: `planner_executor → crux`,
   `aurora_sim → aurora`. Drag if they're wrong.
3. Click **Launch crux** and **Launch aurora**. Both spawn PBS jobs
   via the login node; the dashboard polls qstat until both go R and
   both agents write their `agent_status/<name>.json` file.
4. Once both green, open the Inject panel (right side), pick
   `planner_executor` as recipient, and paste this content:

```json
{
  "spec": {
    "topology": "pcu",
    "backend": "pormake",
    "node_files": ["/lus/eagle/projects/ChemGraph/<user>/venvs/academy-swarm-crux/lib/python3.11/site-packages/pormake/database/bbs/N59.xyz"],
    "edge_files": ["/lus/eagle/projects/ChemGraph/<user>/venvs/academy-swarm-crux/lib/python3.11/site-packages/pormake/database/bbs/E32.xyz"],
    "output_dir": "/lus/eagle/projects/ChemGraph/<user>/mof-out/build"
  },
  "opt_output_json": "/lus/eagle/projects/ChemGraph/<user>/mof-out/opt.json",
  "charges_output_dir": "/lus/eagle/projects/ChemGraph/<user>/mof-out/charges",
  "adsorbate": "H2O",
  "temperature_k": 298.15,
  "pressure_pa": 101325.0
}
```

(Replace `<user>` in every path.)

## What you should see

Observability tab, in order:

1. **planner_executor** on Crux calls `mofforge_build` → CIF written to
   `mof-out/build/`.
2. `run_ase` (UMA optimization) — mock returns in ~2 s, real takes ~15 min.
3. `pacmof2_assign_charges` → charged CIF in `mof-out/charges/`.
4. `transfer_file` — Globus transfer from
   `alcf#dtn_eagle:/ChemGraph/<user>/mof-out/charges/<basename>.cif` to
   `alcf#dtn_flare:/ChemGraph/<user>/inbox-aurora/<basename>.cif`.
5. `send_message(recipient='aurora_sim', ...)`.
6. **aurora_sim** wakes on Aurora: `run_graspa`, `transfer_file` (result
   JSON back to Crux inbox), `send_message` reply, `finish_turn`.
7. **planner_executor** receives the reply. Uptake numbers appear in the
   message content: `{"uptake_mol_kg": <N>, "uptake_cm3_stp_g": <N>,
   "avg_energy_kj_mol": <N>, ...}`.

Final artifacts:

- Crux: `mof-out/{build,charges}/*.cif`, `inbox-crux/*.raspa.json`
- Aurora: `inbox-aurora/*.cif`, `mof-out/graspa/*.raspa.json`

## Endpoint UUIDs

Hardcoded in the missions (both agents get told their own + the peer's):

- `alcf#dtn_eagle` (Crux Eagle DTN, DTN-root = `/ChemGraph/`):
  `05d2c76a-e867-4f67-aa57-76edeb0beda0`
- `alcf#dtn_flare` (Aurora Flare DTN, DTN-root = `/ChemGraph/`):
  `f39a7a0f-5bfc-46ce-9615-ba9f8592814f`

Filesystem paths for local tools use the site's mount (`/eagle/...` on
Crux, `/flare/...` on Aurora); Globus source/dest paths use the
DTN-relative form (`/ChemGraph/...`). The missions explain this to the
LLM.

## Swapping to real gRASPA when it's stable

The mock and real gRASPA share the exact same schema (see
`chemgraph.schemas.graspa_schema.graspa_input_schema` on the
`dev-graspa` branch). To move `mof-crux-aurora-mock` to real gRASPA on
Aurora, just change the `mcp_servers` entry named `mock_sci_aurora`:

```jsonc
{
  "name": "mock_sci_aurora",
  "command": "/flare/ChemGraph/<user>/venvs/academy-swarm/bin/python -m chemgraph.mcp.graspa_mcp_hpc"
}
```

That's it — no mission changes, no schema changes. The mock exists only
to speed up iteration when Aurora is under maintenance or the real
binary isn't installed yet.

## Known limits

- The multi_agent planner sometimes hallucinates tools like `gather` or
  `submit_result`; the runtime returns errors and the planner moves on.
  Not a blocker, just noise in the event stream.
- After receiving the sim's reply, the planner does not always call
  `finish_turn` on its own; PBS walltime terminates the run cleanly.
  Cosmetic.
