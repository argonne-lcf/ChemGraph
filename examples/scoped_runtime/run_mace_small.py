"""Review and run a scoped H2 optimization with MACE-MP small.

Run with python -u examples/scoped_runtime/run_mace_small.py. MACE downloads
the small weights automatically on first use and reuses its cache afterward.
Each run keeps its inputs, operation records, and artifacts in a temp folder.
"""

import argparse
import asyncio
import json
import math
import tempfile
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="codex:gpt-5.6-luna",
        help="Configured ChemGraph LLM (default: codex:gpt-5.6-luna).",
    )
    return parser.parse_args()


def progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def show_event(event, payload):
    if event == "llm_decision":
        detail = ", ".join(call["name"] for call in payload.get("tool_calls", []))
    elif event == "tool_call_started":
        detail = f"{payload.get('tool_name')}: {payload.get('arguments')}"
    elif event == "tool_call_finished":
        detail = str(payload.get("result", ""))
    else:
        detail = str(payload.get("error", ""))
    progress(f"{event}: {detail[:800]}" if detail else event)


async def main(model_name):
    root = Path(tempfile.mkdtemp(prefix="chemgraph-scoped-e2e-")).resolve()
    progress(f"Run files: {root}")
    progress("Loading MACE dependencies...")
    try:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint

        progress("Finding MACE-MP small in the cache; downloading if needed...")
        # Scoped approval includes a weights checksum, so provision the model
        # before preparation, using the same downloader as MACE's calculator.
        model_file = Path(await asyncio.to_thread(
            download_mace_mp_checkpoint, model="small",
        )).resolve(strict=True)
    except ImportError as exc:
        progress(f"MACE dependencies are unavailable: {exc}")
        progress("Install ChemGraph in this environment with python -m pip install -e .")
        return 1
    except (OSError, RuntimeError) as exc:
        progress(f"Could not obtain MACE-MP small weights: {exc}")
        progress("Check network access and permissions for the MACE cache, then retry.")
        return 1
    progress(f"MACE-MP small weights: {model_file}")
    progress("Loading ChemGraph dependencies...")

    from langgraph.checkpoint.memory import InMemorySaver

    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.agent.main_session import MainAgentSession
    from chemgraph.registry.tools import ToolRegistry
    from chemgraph.runtime import (
        ExecutionPolicy, FileOperationStore, OperationJournal, RunContext,
        artifact_file, result_artifacts, run_context,
    )

    policy = ExecutionPolicy(
        calculators={"mace_mp"},
        model_roots=[model_file.parent],
        model_resources={"mace_mp:small": model_file},
        limits={"max_steps": 200, "max_atoms": 50},
    )
    journal = OperationJournal(FileOperationStore(root / "ops"))
    await journal.recover()
    ctx = RunContext(
        owner="me", thread_id="t1", task_id="task-1",
        workspace=root / "ws", policy=policy, journal=journal,
    )
    input_file = ctx.workspace / "h2.xyz"
    input_file.write_text("2\nH2 test structure\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    progress(f"Input structure: {input_file}")
    progress(f"Creating the agent with {model_name}...")
    agent = ChemGraph(
        workflow_type="main_agent", model_name=model_name,
        approval_mode="scoped", subagent_names=[], discover_skills=False,
        enable_memory=False, log_dir=str(root / "logs"),
        checkpointer=InMemorySaver(),
        tool_registry=ToolRegistry(
            spec for spec in ToolRegistry().specs() if spec.name in policy.tools
        ),
    )
    session = MainAgentSession(
        agent.workflow, thread_id=ctx.thread_id,
        session_metadata=agent.main_agent_metadata, on_event=show_event,
    )
    params = {
        "input_structure_file": str(input_file),
        "output_results_file": "h2_opt.json",
        "calculator": {
            "calculator_type": "mace_mp", "model": "small",
            "device": "cpu", "default_dtype": "float64",
        },
        "driver": "opt", "steps": 200, "fmax": 0.01,
    }
    progress("Starting the agent turn; waiting for a prepared action to review...")
    with run_context(ctx):
        turn = await session.run(
            f"Optimize H2 with MACE-MP small. The input already exists at {input_file}. "
            f"The scoped science workspace is {ctx.workspace}. "
            "The ls/read_file tools inspect checkpoint files and do not expose "
            "this host workspace. Use native chemistry tools for the structure. "
            "Load run_ase using load_tools(names=['run_ase']), then call run_ase "
            f"with params={json.dumps(params)}. "
            "The operator has bound model='small' to cached weights; retain "
            "that selection. Report the saved final energy and convergence."
        )

    while turn.status == "waiting_for_user":
        answers = {}
        for item in turn.interrupts:
            review = item.payload
            if not isinstance(review, dict) or review.get("kind") != "approval":
                raise RuntimeError(f"Unexpected interrupt: {review!r}")
            print(f"\nREVIEW: {review['tool']}", flush=True)
            print("Operation:", review["operation_id"], flush=True)
            print("Digest:", review["digest"], flush=True)
            print(json.dumps(review["prepared"], indent=2), flush=True)
            answer = await asyncio.to_thread(input, "Approve this action? [y/N] ")
            approved = answer.strip().lower() in {"y", "yes"}
            feedback = None
            if not approved:
                feedback = (await asyncio.to_thread(
                    input, "Rejection feedback (Enter to reject without a revision request): ",
                )).strip() or None
            await journal.decide(
                review["operation_id"], digest=review["digest"], approve=approved,
                decided_by=ctx.owner, expires_at=time.time() + 600, message=feedback,
            )
            answers[item.id] = "decided"
        progress("Resuming the agent with the recorded decisions...")
        # The journal decision authorizes execution; this value only resumes it.
        with run_context(ctx):
            turn = await session.resume(answers)

    print("\nASSISTANT:", turn.assistant_response, flush=True)
    artifacts = result_artifacts(turn.state, ctx.task_id)
    records = await journal.store.records(thread_id=ctx.thread_id)
    finalized_calls = {
        record["tool_call_id"] for record in records
        if record["tool"] == "run_ase" and record["state"] == "finalized"
    }
    for record in records:
        progress(f"Operation {record['tool_call_id']}: {record['tool']} -> {record['state']}")
        if record.get("error"):
            progress(f"Preparation error: {record['error']}")

    found_result = False
    succeeded = False
    for artifact in artifacts:
        path = artifact_file(ctx.workspace, artifact)
        print(f"ARTIFACT: {artifact['source_path']} -> {path}", flush=True)
        if artifact["tool_call_id"] not in finalized_calls or path.suffix != ".json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        simulation = result.get("simulation_input", {})
        if simulation.get("driver") != "opt":
            continue
        found_result = True
        energy = result.get("potential_energy")
        calculator = simulation.get("calculator", {})
        valid_energy = type(energy) in (int, float) and math.isfinite(energy)
        selected_model = calculator.get("model")
        correct_model = (
            calculator.get("calculator_type") == "mace_mp"
            and selected_model == str(model_file)
        )
        print("ENERGY:", energy, result.get("energy_unit", "eV"), flush=True)
        print("CONVERGED:", result.get("converged"), flush=True)
        print("MODEL:", selected_model, flush=True)
        if result.get("error"):
            print("CALCULATION ERROR:", result["error"], flush=True)
        succeeded |= (
            result.get("success") is True and result.get("converged") is True
            and valid_energy and correct_model
        )
    print("\nWORKSPACE:", root, flush=True)
    if not found_result:
        progress("No optimization result was produced. The calculation test did not pass.")
        return 1
    if not succeeded:
        progress("A result was saved, but a converged MACE-MP small optimization was not verified.")
        return 1
    progress("Verified a finalized, converged MACE-MP small optimization with a finite energy.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(asyncio.run(main(args.model)))
