"""Run a traditional ChemGraph workflow and write dashboard artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import threading
import traceback
from pathlib import Path

from chemgraph.academy.core.lm import load_lm_config
from chemgraph.observability.workflow_runner import run_observed_chemgraph_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local traditional ChemGraph workflow and emit event artifacts "
            "that can be visualized by the ChemGraph dashboard."
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--workflow-type", default="single_agent")
    parser.add_argument("--return-option", choices=["last_message", "state"], default="state")
    parser.add_argument("--recursion-limit", type=int, default=50)
    parser.add_argument("--lm-config")
    parser.add_argument("--model-name")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--argo-user")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing local dashboard run directory.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print the full workflow result JSON to stdout.",
    )
    return parser.parse_args()


def _prepare_run_dir(path: Path, *, overwrite: bool) -> None:
    existing_artifacts = [
        path / "events.jsonl",
        path / "status.json",
        path / "manifest.json",
        path / "result.json",
        path / "chemgraph_workflows",
    ]
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if overwrite:
        _clear_run_dir(path)
    elif any(item.exists() for item in existing_artifacts):
        raise RuntimeError(
            f"Run directory already contains dashboard artifacts: {path}\n"
            "Use a new --run-dir, run chemgraph-dashboard to view the "
            "existing run, or pass --overwrite to replace it.",
        )
    path.mkdir(parents=True, exist_ok=True)


def _clear_run_dir(path: Path) -> None:
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


async def _run(args: argparse.Namespace) -> dict:
    model_name = args.model_name
    base_url = args.base_url
    api_key = args.api_key
    argo_user = args.argo_user
    if args.lm_config:
        settings = load_lm_config(args.lm_config)
        model_name = model_name or settings.model
        base_url = base_url or settings.base_url
        api_key = api_key or settings.api_key
        argo_user = argo_user or settings.user

    return await run_observed_chemgraph_workflow(
        query=args.query,
        run_dir=Path(args.run_dir),
        workflow_type=args.workflow_type,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        argo_user=argo_user,
        return_option=args.return_option,
        recursion_limit=args.recursion_limit,
        write_run_files=True,
    )


def _print_result_summary(*, result: dict, run_dir: Path, json_output: bool) -> None:
    result_path = run_dir / "result.json"
    print(
        "ChemGraph workflow completed.\n"
        f"  status: {result.get('status')}\n"
        f"  workflow: {result.get('workflow_type')}\n"
        f"  span: {result.get('span_id')}\n"
        f"  result: {result_path}",
        flush=True,
    )
    if json_output:
        print(json.dumps(result, indent=2, default=str), flush=True)


def _run_and_report(args: argparse.Namespace, *, run_dir: Path) -> None:
    try:
        result = asyncio.run(_run(args))
    except Exception:  # noqa: BLE001 - surface background workflow failures
        print("ChemGraph workflow failed. See status.json/events.jsonl if present.", flush=True)
        traceback.print_exc()
        return
    _print_result_summary(
        result=result,
        run_dir=run_dir,
        json_output=args.json_output,
    )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    _prepare_run_dir(run_dir, overwrite=args.overwrite)
    args.run_dir = str(run_dir)
    if args.serve:
        from chemgraph.academy.dashboard import serve_dashboard

        thread = threading.Thread(
            target=_run_and_report,
            kwargs={"args": args, "run_dir": run_dir},
            name="chemgraph-dashboard-workflow",
            daemon=True,
        )
        thread.start()
        return serve_dashboard(
            run_dir=run_dir,
            host=args.host,
            port=args.port,
        )

    result = asyncio.run(_run(args))
    _print_result_summary(
        result=result,
        run_dir=run_dir,
        json_output=args.json_output,
    )
    print(
        "Dashboard command: "
        f"chemgraph-dashboard --run-dir {run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
