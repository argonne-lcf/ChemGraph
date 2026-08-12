"""Run the ALCF IRI Facility API workflow through the ChemGraph agent.

Uses the ``single_agent_iri`` workflow, which binds seven category
dispatcher tools (facility / status / account / compute / filesystem /
task / auth) covering ALCF's IRI REST API (https://api.alcf.anl.gov).
The agent discovers per-action schemas on demand via `list_actions`
and `describe`.

The default PROMPT below hits only public endpoints (machine status),
so it works without any auth token -- good smoke test. See the README
for queries that need $ALCF_API_TOKEN.

Prerequisites:
  - An LLM provider key (e.g. OPENAI_API_KEY)
  - (Optional) $ALCF_API_TOKEN for auth-gated actions -- see README.

Usage:
  export OPENAI_API_KEY="your_key"
  python run_chemgraph.py
"""

import asyncio

from chemgraph.agent.llm_agent import ChemGraph

MODEL_NAME = "gpt-4o-mini"

# Try changing this to any of the example queries in the README.
PROMPT = "Which ALCF machines are currently up, and which are down or in unknown state?"


async def main():
    cg = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type="single_agent_iri",
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}\n")
    result = await cg.run(PROMPT)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
