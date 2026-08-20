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

## Two tool sets

`single_agent_iri` is a single graph that can be bound to either of two
shipped tool sets:

| Tool set | Import from | Tools | When to use |
|---|---|---:|---|
| **`ALCF_IRI_FLAT_TOOLS`** (default) | `chemgraph.tools.alcf_iri_flat_tools` | 43 | One tool per IRI endpoint. Higher judge score in our eval; recommended default. |
| **`ALCF_IRI_CATEGORY_TOOLS`** | `chemgraph.tools.alcf_iri_tools` | 7 | Category dispatchers with a `list_actions`/`describe` discovery protocol. ~3× smaller upfront schema surface; useful when context is tight or the model handles many-tool prompts poorly. |

Pick one at construction time; nothing else changes:

```python
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.tools.alcf_iri_flat_tools import ALCF_IRI_FLAT_TOOLS
# from chemgraph.tools.alcf_iri_tools import ALCF_IRI_CATEGORY_TOOLS

cg = ChemGraph(
    model_name="gpt-4o-mini",
    workflow_type="single_agent_iri",
    tools=ALCF_IRI_FLAT_TOOLS,  # or ALCF_IRI_CATEGORY_TOOLS
)
```

The graph auto-selects a matching system prompt when you don't pass one
(`alcf_iri_flat_prompt` for flat, `alcf_iri_prompt` for category).

## What's here
- `run_chemgraph.py` — runs one query end-to-end through **both** tool sets
  so you can see the swap pattern.
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

**Flat** (default): one `@tool` per IRI endpoint. ~43 tools, ~5k tokens of
tool-schema in the prompt, but each schema is small and self-documenting
(`alcf_status_list_resources` has just the args it needs). No discovery
turn; the model picks the tool by name.

**Category**: 7 dispatcher tools, each taking an `action` enum plus optional
`params`. Cuts prompt schema footprint ~3× at rest, at the cost of one
extra LLM turn on cold action lookups (`list_actions` / `describe`). Useful
when the flat 43-tool surface is prohibitive.

## Also available as an MCP server

The same tools ship as a standalone MCP server for use with
non-LangChain clients (Claude Desktop, other agent frameworks,
`main_agent`'s MCP wiring). Both tool-set variants are supported:

```bash
python -m chemgraph.mcp.alcf_iri_mcp                                 # flat, stdio
python -m chemgraph.mcp.alcf_iri_mcp --variant category               # category
python -m chemgraph.mcp.alcf_iri_mcp --transport streamable_http --port 9010
```

Env-var equivalent for MCP client configs that only pass env:
`CHEMGRAPH_IRI_MCP_VARIANT=flat|category`. Default is flat.

Same auth flow (`$ALCF_API_TOKEN` -> on-disk Globus cache -> interactive
re-auth via `alcf_auth_start_reauth` / `alcf_auth_complete_reauth` for
flat, or `alcf_auth(action='start_reauth' | 'complete_reauth')` for
category). Same `$ALCF_IRI_ALLOW_UNSAFE=1` gate for write actions. A
capability card for skill-routing agents (e.g. `main_agent`) lives at
`src/chemgraph/skills/alcf_iri.md`.

## References
- ALCF IRI docs: https://docs.alcf.anl.gov/services/iri-api/
- OpenAPI spec: https://api.alcf.anl.gov/openapi.json
- ALCF's token helper: https://github.com/argonne-lcf/alcf-facility-api-token
