"""One bounded run per spawned process; HTTP connections never own execution."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

from chemgraph.api.settings import Settings
from chemgraph.api.store import Store

logger = logging.getLogger(__name__)


def execute_run(
    root: str, run_id: str, settings_data: dict, parent_pid: int | None = None
):
    """Spawn entry point. Only primitive values cross the process boundary."""
    if parent_pid is not None:
        import threading

        def watch_parent():
            while os.getppid() == parent_pid:
                time.sleep(1)
            # A killed API cannot join its workers. Stop this worker rather than
            # leaving an orphan calculation running after restart recovery.
            os._exit(1)

        threading.Thread(target=watch_parent, daemon=True).start()
    settings = Settings(**settings_data)
    store = Store(Path(root))
    run = store.one("SELECT * FROM runs WHERE id=?", (run_id,))
    session = store.one("SELECT * FROM sessions WHERE id=?", (run["session_id"],))
    workspace = store.session_dir(session["id"])
    turn_dir = workspace / run_id
    turn_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(turn_dir)
    os.environ["CHEMGRAPH_LOG_DIR"] = str(turn_dir)
    store.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
    store.event(run_id, "status", {"status": "running"})

    async def ask(question):
        question_id = str(uuid4())
        with store.connect() as db:
            db.execute(
                "UPDATE runs SET status='waiting_for_input', question_id=?, question=?, answer=NULL WHERE id=?",
                (question_id, question, run_id),
            )
        store.event(run_id, "status", {"status": "waiting_for_input"})
        while True:
            row = store.one("SELECT answer FROM runs WHERE id=?", (run_id,))
            if row["answer"] is not None:
                store.execute(
                    "UPDATE runs SET status='running', question_id=NULL, question=NULL WHERE id=?",
                    (run_id,),
                )
                store.event(run_id, "status", {"status": "running"})
                return row["answer"]
            await asyncio.sleep(0.2)

    def event(kind, payload):
        # Keep raw provider responses, exception reprs, tool arguments and paths
        # off the browser event stream. The API exposes registered artifacts.
        if kind in {
            "tool_call_started",
            "tool_call_finished",
            "tool_call_failed",
            "llm_call_started",
            "llm_call_finished",
        }:
            store.event(
                run_id, "progress", {"kind": kind, "tool": payload.get("tool_name")}
            )

    async def run_agent():
        import json
        from chemgraph.agent.llm_agent import ChemGraph
        from chemgraph.api.chemistry import web_tools
        from chemgraph.memory.store import SessionStore

        provider = settings.providers[session["model"]]
        api_key = os.getenv(provider.api_key_env) if provider.api_key_env else None
        if provider.api_key_env and not api_key:
            raise ValueError("The selected provider is not configured on the server.")
        memory = SessionStore(str(store.root / "memory" / f"{session['id']}.db"))
        previous = memory.get_session(session["id"])
        agent = ChemGraph(
            model_name=provider.model,
            base_url=provider.base_url,
            api_key=api_key,
            argo_user=provider.argo_user,
            workflow_type=session["workflow"],
            return_option="last_message",
            human_supervised=True,
            human_input_handler=ask,
            on_event=event,
            session_store=memory,
            log_dir=str(turn_dir),
            tools=web_tools(workspace, settings.calculators),
        )
        agent.uuid = session["id"]
        agent._session_created = previous is not None
        query = run["query"]
        attachment_ids = json.loads(run["attachments"])
        if attachment_ids:
            paths = [
                store.file_path(store.one("SELECT * FROM files WHERE id=?", (file_id,)))
                for file_id in attachment_ids
            ]
            query += (
                "\n\nAttached files (use these exact workspace paths):\n"
                + "\n".join(str(p) for p in paths)
            )
        query += f"\n\nApproved calculators: {', '.join(settings.calculators)}. Write new outputs in {turn_dir}."
        result = await agent.run(query, resume_from=session["id"] if previous else None)
        content = getattr(result, "content", result)
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content)

    try:
        if settings.demo:
            # Explicit development-only demo: real EMT, no model or network.
            text = asyncio.run(demo_run(turn_dir, run["query"], ask, event))
        else:
            text = asyncio.run(run_agent())
        from ui.artifacts import classify_artifacts

        kinds = classify_artifacts([p.name for p in turn_dir.iterdir() if p.is_file()])
        by_name = {name: kind for kind, names in kinds.items() for name in names}
        for path in sorted(turn_dir.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower()
                in {".xyz", ".pdb", ".cif", ".traj", ".json", ".csv", ".png", ".html"}
            ):
                store.register_file(
                    session["id"], path, by_name.get(path.name, "data"), run_id
                )
        # Display transcript paths as opaque relative names, not host paths.
        text = text.replace(str(workspace) + "/", "")
        store.finish(run_id, "completed", text=text)
    except Exception:
        logger.exception("Web run %s failed", run_id)
        store.finish(
            run_id,
            "failed",
            error="The calculation failed. Check the server log using this run ID, then retry.",
        )


async def demo_run(turn_dir, query, ask, event):
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.io import write
    from ase.optimize import BFGS

    if "confirm" in query.lower():
        await ask("Optimize this copper dimer with the EMT calculator?")
    event("tool_call_started", {"tool_name": "run_ase"})
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.5]])
    atoms.calc = EMT()
    with BFGS(
        atoms, trajectory=str(turn_dir / "copper_opt.traj"), logfile=None
    ) as optimizer:
        optimizer.run(fmax=0.05, steps=10)
    write(str(turn_dir / "copper.xyz"), atoms)
    event("tool_call_finished", {"tool_name": "run_ase"})
    await asyncio.sleep(0.2)
    return f"Demo calculation complete. The copper dimer has a potential energy of **{atoms.get_potential_energy():.4f} eV** using EMT. Inspect the structure and optimization trajectory in Results."


class Supervisor:
    """Bounded process scheduler with durable queued work and explicit recovery."""

    def __init__(self, store: Store, settings: Settings):
        self.store, self.settings = store, settings
        self.children = {}
        self.task = None

    async def start(self):
        # The service is intentionally single-process. Prevent a second API
        # instance from interrupting live runs or executing queued work twice.
        import fcntl

        self.lock = open(self.store.root / "server.lock", "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError(
                "Only one API process may use this data directory."
            ) from None
        self.store.recover()
        self.task = asyncio.create_task(self.loop())

    async def stop(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Web scheduler stopped unexpectedly")
        for run_id, (process, _) in self.children.items():
            await asyncio.to_thread(self.terminate, process)
            self.store.finish(
                run_id,
                "interrupted",
                error="The server stopped. You can retry this request.",
            )
        self.lock.close()

    @staticmethod
    def terminate(process):
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()

    async def loop(self):
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        while True:
            for run_id, (process, started) in list(self.children.items()):
                timed_out = time.monotonic() - started > self.settings.run_timeout
                if not process.is_alive() or timed_out:
                    await asyncio.to_thread(self.terminate, process)
                    self.store.finish(
                        run_id,
                        "interrupted",
                        error="The worker stopped or exceeded its time limit. You can retry.",
                    )
                    del self.children[run_id]
            for row in self.store.rows(
                "SELECT id FROM runs WHERE status='queued' ORDER BY created"
            ):
                if len(self.children) >= self.settings.max_workers:
                    break
                if row["id"] in self.children:
                    continue
                process = context.Process(
                    target=execute_run,
                    args=(
                        str(self.store.root),
                        row["id"],
                        self.settings.model_dump(),
                        os.getpid(),
                    ),
                    daemon=True,
                )
                try:
                    process.start()
                except Exception:
                    logger.exception("Could not start web worker %s", row["id"])
                    self.store.finish(
                        row["id"],
                        "failed",
                        error="The server could not start a calculation worker. Please retry.",
                    )
                    continue
                self.children[row["id"]] = (process, time.monotonic())
            await asyncio.sleep(0.2)
