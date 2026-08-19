"""Run the ALCF IRI Facility API workflow through the ChemGraph agent.

Uses the ``single_agent_iri`` workflow, which can be bound to either of
two shipped tool sets (both cover ALCF's IRI REST API at
https://api.alcf.anl.gov):

  * ALCF_IRI_FLAT_TOOLS      -- 43 direct tool wrappers (default;
                                 higher judge score on our eval).
  * ALCF_IRI_CATEGORY_TOOLS  -- 7 category dispatcher tools that use a
                                 discovery protocol (``list_actions`` /
                                 ``describe``). Smaller upfront schema
                                 surface; useful under tight context limits.

This example runs the same query through both tool sets so you can see
the tool-list swap pattern. The workflow_type stays the same
(``single_agent_iri``); only ``tools=`` changes.

The default PROMPT hits only public endpoints (machine status), so it
works without any auth token -- good smoke test. See the README for
queries that need $ALCF_API_TOKEN.

Prerequisites:
  - An LLM provider key (e.g. OPENAI_API_KEY)
  - (Optional) $ALCF_API_TOKEN for auth-gated actions -- see README.

Usage:
  export OPENAI_API_KEY="your_key"
  python run_chemgraph.py
"""

import asyncio

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.tools.alcf_iri_tools import ALCF_IRI_CATEGORY_TOOLS
from chemgraph.tools.alcf_iri_flat_tools import ALCF_IRI_FLAT_TOOLS

MODEL_NAME = "gpt-4o-mini"

# Try changing this to any of the example queries in the README.
PROMPT = "Which ALCF machines are currently up, and which are down or in unknown state?"


async def main():
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}\n")

    for label, tools in [
        ("flat (43 direct tools, default)",       ALCF_IRI_FLAT_TOOLS),
        ("category (7 tools + discovery)",         ALCF_IRI_CATEGORY_TOOLS),
    ]:
        print(f"\n===== tool set: {label} =====")
        cg = ChemGraph(
            model_name=MODEL_NAME,
            workflow_type="single_agent_iri",
            tools=tools,
        )
        result = await cg.run(PROMPT)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
