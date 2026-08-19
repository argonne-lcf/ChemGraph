"""Explore ChemGraph's tool and worker-agent registries.

The default run needs no model or provider credentials. It lists registered
entries, checks optional runtime requirements, resolves a small tool set, and
invokes the calculator directly.

Pass ``--model`` (with that provider configured) to also construct
``single_agent`` as a standalone graph and as a ``CompiledSubAgent``.
Constructing the graphs does not call the model.

Examples
--------
python examples/registries/run_registries.py
python examples/registries/run_registries.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse

from chemgraph.registry import AgentRegistry, ToolRegistry


DEMO_TOOL_NAMES = (
    "molecule_name_to_smiles",
    "smiles_to_coordinate_file",
    "calculator",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional ChemGraph model name used to construct example worker "
            "graphs, such as gpt-4o-mini."
        ),
    )
    return parser.parse_args()


def show_tool_registry(registry: ToolRegistry) -> list:
    """Inspect tools and return a small, dependency-safe tool set."""
    print("Registered tools:")
    for spec in registry.specs():
        status = registry.availability(spec.name)
        availability = "available" if status.available else "unavailable"
        print(f"  {spec.name:26} {availability:11} tags={sorted(spec.tags)}")
        for issue in status.issues:
            print(f"    - {issue}")

    tools = registry.resolve(DEMO_TOOL_NAMES, require_available=True)
    print("\nResolved demo tools:", ", ".join(tool.name for tool in tools))

    calculator = registry.get("calculator", require_available=True)
    result = calculator.invoke({"expression": "2 * pi + 5"})
    print("Calculator result for '2 * pi + 5':", result)
    return tools


def show_agent_registry(registry: AgentRegistry) -> None:
    """Inspect worker graphs without importing or constructing them."""
    print("\nRegistered worker agents:")
    for spec in registry.specs():
        status = registry.availability(spec.name)
        availability = "available" if status.available else "unavailable"
        aliases = f" aliases={list(spec.aliases)}" if spec.aliases else ""
        print(f"  {spec.name:26} {availability:11}{aliases}")
        for issue in status.issues:
            print(f"    - {issue}")

    print("\nAlias resolution:")
    print("  python_repl ->", registry.resolve_name("python_repl"))
    print("  graspa_agent ->", registry.resolve_name("graspa_agent"))
    print("  iri ->", registry.resolve_name("iri"))
    print("  main_agent is intentionally not registered")


def construct_worker_examples(
    registry: AgentRegistry,
    *,
    model_name: str,
    tools: list,
) -> None:
    """Construct a worker in standalone and parent-checkpointed forms."""
    from chemgraph.models.loader import load_chat_model

    llm = load_chat_model(model_name)

    standalone = registry.build(
        "single_agent",
        llm=llm,
        tools=tools,
    )
    print("\nStandalone single_agent:")
    print("  graph type:", type(standalone).__name__)
    print("  checkpointer:", type(standalone.checkpointer).__name__)

    subagent = registry.as_subagent(
        "single_agent",
        llm=llm,
        tools=tools,
    )
    print("\nCompiledSubAgent (not bound to main_agent):")
    print("  name:", subagent["name"])
    print("  description:", subagent["description"])
    print("  inherits parent checkpointer:", subagent["runnable"].checkpointer is None)


def main() -> int:
    """Run the registry demonstration."""
    args = parse_args()
    tools = show_tool_registry(ToolRegistry())

    agent_registry = AgentRegistry()
    show_agent_registry(agent_registry)

    if args.model:
        construct_worker_examples(
            agent_registry,
            model_name=args.model,
            tools=tools,
        )
    else:
        print("\nPass --model to construct standalone and subagent worker graphs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
