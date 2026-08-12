# ALCF IRI Facility API with ChemGraph

Runs the **`single_agent_iri`** workflow that gives an agent access to ALCF's
IRI Facility API (https://api.alcf.anl.gov). The agent can answer questions
about ALCF machine state, project allocations, PBS jobs, and (once ALCF
provisions your identity) remote filesystem contents — all against the real
API, no ssh required.

You can run it from the CLI:
```bash
chemgraph -q "Which ALCF machines are currently up?" -w iri
```
or pick **single_agent_iri** as the workflow in the Streamlit UI (`streamlit run src/ui/app.py`).

## What's here
- `run_chemgraph.py` — runs one query end-to-end and prints the answer.
- (this README)

## Setup

### 1. Install ChemGraph
```bash
pip install -e .
```

`globus-sdk` is a transitive dependency and gets pulled in automatically.

### 2. Get an ALCF IRI access token

Two paths — pick one.

**A. In-chat (recommended for first use).** Ask the agent to run `alcf_auth`:
```bash
chemgraph -w iri -q "Authenticate to ALCF."
```
The agent returns a Globus URL. Open it, sign in with your ALCF-linked identity
(`<your-username>@alcf.anl.gov`), and paste the resulting code back into the
chat. Tokens cache for 30 days, silently refresh thereafter.

**B. ALCF's CLI helper** (equivalent, terminal-based):
```bash
mkdir -p ~/tools
curl -o ~/tools/alcf_facility_api_globus_token.py \
    https://raw.githubusercontent.com/argonne-lcf/alcf-facility-api-token/refs/heads/main/alcf_facility_api_globus_token.py
python ~/tools/alcf_facility_api_globus_token.py authenticate
export ALCF_API_TOKEN=$(python ~/tools/alcf_facility_api_globus_token.py get_access_token)
```

Either path writes to the same on-disk token cache
(`~/.globus/app/8b84fc2d-.../alcf_facility_api_app/tokens.json`),
so they interoperate.

### 3. LLM backend

Whatever ChemGraph supports (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`ARGO_USER` + argo-shim, etc.). See top-level README for options.

## Run
```bash
python run_chemgraph.py
```

## Try your own
- Public endpoints (machine status, incidents, events, facility metadata)
  work without any auth token.
- Authenticated endpoints (projects, allocations, PBS jobs) need step 2 above.
- Write actions (submit_job, cancel_job, mkdir, rm, chmod, ...) additionally
  require `export ALCF_IRI_ALLOW_UNSAFE=1`. Use with care.

## Example queries

**Machine status (one hop, no auth)**
- *Which ALCF machines are currently up?*
- *Is Aurora in maintenance?*

**Allocations (two hop)**
- *Find the ChemGraph project and list all its allocations across every machine.*
- *Which of my allocations is closest to being exhausted?*

**Jobs**
- *Show the status of Crux job 246115 and describe what happened.*
- *List my last 5 completed jobs on Polaris.*

**Multi-tool health check (three hop)**
- *For every compute machine that's up, tell me whether my project has
  allocation left and how many jobs I have in the queue.*

**Job submission (requires `ALCF_IRI_ALLOW_UNSAFE=1`)**
- *Submit a smoke-test job on Crux: run `/bin/bash -lc 'echo hello; sleep 10'`
  under the ChemGraph account, 5-minute duration, 1 node in the debug queue,
  stdout+stderr to `/home/<user>/iri-demo.{out,err}`, filesystems `home:eagle`.
  Then keep checking status until it finishes.*

## Design at a glance

Instead of binding one `@tool` per endpoint (~43 tools, ~8k tokens of schemas
upfront), `single_agent_iri` bundles the endpoints into **7 category tools**
(facility / status / account / compute / filesystem / task / auth), each
accepting an `action` enum. The LLM discovers per-action schemas on demand
via `list_actions` / `describe`. Trades one extra LLM turn on cold action
lookups for a ~4x reduction in the tool-schema prompt footprint.

A flat-tool baseline is also shipped as `single_agent_iri_flat` for
head-to-head comparison (see `notebooks/iri_benchmark.ipynb`).

## References
- ALCF IRI docs: https://docs.alcf.anl.gov/services/iri-api/
- OpenAPI spec: https://api.alcf.anl.gov/openapi.json
- ALCF's token helper: https://github.com/argonne-lcf/alcf-facility-api-token
