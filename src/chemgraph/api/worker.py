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
from chemgraph.api.providers import credentials, session_status
from chemgraph.api.errors import public_error, RunStopped, StorageLimit

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
    if not store.execute(
        "UPDATE runs SET status='running' WHERE id=? AND status IN ('queued','running')",
        (run_id,),
    ):
        return
    store.event(run_id, "status", {"status": "running"})

    def guard():
        row = store.one("SELECT status FROM runs WHERE id=?", (run_id,))
        if row["status"] not in {"running", "waiting_for_input"}:
            raise RunStopped()
        if store.owner_usage(session["owner"]) > settings.user_storage_limit:
            raise StorageLimit()

    secret_values = []
    for provider in settings.providers.values():
        try:
            key = credentials(provider).api_key
            if key and not key.startswith("dummy_"):
                secret_values.append(key)
        except ValueError:
            pass

    def safe_text(value):
        text = str(value).replace(str(workspace) + "/", "")
        for secret in secret_values:
            text = text.replace(secret, "[redacted]")
        return text

    async def ask(question):
        question_id = str(uuid4())
        with store.transaction() as db:
            db.execute(
                "UPDATE runs SET status='waiting_for_input', question_id=?, question=?, answer=NULL,waiting_since=? WHERE id=? AND status='running'",
                (question_id, safe_text(question), time.time(), run_id),
            )
        store.event(run_id, "status", {"status": "waiting_for_input"})
        while True:
            guard()
            row = store.one("SELECT answer FROM runs WHERE id=?", (run_id,))
            if row["answer"] is not None:
                store.execute(
                    "UPDATE runs SET status='running', question_id=NULL, question=NULL,waiting_since=NULL WHERE id=? AND status='waiting_for_input'",
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
                run_id,
                "progress",
                {
                    "kind": kind,
                    "tool": safe_text(payload["tool_name"])
                    if payload.get("tool_name")
                    else None,
                },
            )

    async def run_agent():
        import json
        from chemgraph.agent.llm_agent import ChemGraph
        from chemgraph.api.chemistry import web_tools
        from chemgraph.api.memory import WebSessionStore

        provider = settings.providers[session["model"]]
        if not session_status(settings, session)["configured"]:
            raise ValueError("The selected provider is not configured on the server.")
        shared = credentials(provider)
        memory = WebSessionStore(
            str(store.root / "memory" / f"{session['id']}.db"),
            run["query"],
            settings.context_limit,
        )
        with memory._connect() as db:
            previous = db.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session["id"],)
            ).fetchone()
        diagnostic_dir = store.root / "diagnostics" / session["id"] / run_id
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        agent = ChemGraph(
            model_name=provider.model,
            base_url=provider.base_url,
            api_key=shared.api_key,
            argo_user=shared.argo_user,
            model_timeout=settings.provider_timeout,
            workflow_type=session["workflow"],
            return_option="last_message",
            human_supervised=True,
            human_input_handler=ask,
            on_event=event,
            session_store=memory,
            log_dir=str(diagnostic_dir),
            tools=web_tools(
                workspace, settings.calculators, guard=guard, publish=publish
            ),
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

    published = set()

    def publish(path):
        path = Path(path).resolve()
        path.relative_to(turn_dir.resolve())
        if path.is_file() and not path.is_symlink():
            published.add(path)

    try:
        if settings.demo:
            # Explicit development-only demo: real EMT, no model or network.
            text = asyncio.run(demo_run(turn_dir, run["query"], ask, event))
            for name in ("copper.xyz", "copper_opt.traj"):
                publish(turn_dir / name)
        else:
            text = asyncio.run(run_agent())
        from ui.artifacts import classify_artifacts

        guard()
        kinds = classify_artifacts([p.name for p in published])
        by_name = {name: kind for kind, names in kinds.items() for name in names}
        for path in sorted(published):
            store.register_file(
                session["id"], path, by_name.get(path.name, "data"), run_id
            )
        # Display transcript paths as opaque relative names, not host paths.
        store.finish(run_id, "completed", text=safe_text(text))
    except RunStopped:
        pass  # The supervisor finalizes cancellation after joining the process.
    except Exception as exc:
        logger.exception("Web run %s failed", run_id)
        code, message = public_error(exc)
        store.finish(
            run_id,
            "failed",
            error=message,
            error_code=code,
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
                stopped=True,
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
        last_cleanup = 0
        while True:
            if time.monotonic() - last_cleanup > 3600:
                await asyncio.to_thread(
                    self.store.cleanup, self.settings.retention_days
                )
                last_cleanup = time.monotonic()
            for run_id, (process, started) in list(self.children.items()):
                row = self.store.one(
                    "SELECT r.*,s.owner FROM runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=?",
                    (run_id,),
                )
                timed_out = time.monotonic() - started > self.settings.run_timeout
                waiting_expired = (
                    row["waiting_since"] is not None
                    and time.time() - row["waiting_since"]
                    > self.settings.clarification_timeout
                )
                cancelled = row["status"] == "cancelling"
                over_quota = (
                    await asyncio.to_thread(self.store.owner_usage, row["owner"])
                    > self.settings.user_storage_limit
                )
                if (
                    row["status"] in {"completed", "failed", "interrupted", "cancelled"}
                    or not process.is_alive()
                    or timed_out
                    or waiting_expired
                    or cancelled
                    or over_quota
                ):
                    await asyncio.to_thread(self.terminate, process)
                    self.store.finish(
                        run_id,
                        "cancelled" if cancelled else "interrupted",
                        error="The run was cancelled."
                        if cancelled
                        else "The clarification request expired."
                        if waiting_expired
                        else "Your workspace storage limit was reached."
                        if over_quota
                        else "The worker stopped or exceeded its time limit. You can retry.",
                        error_code="cancelled"
                        if cancelled
                        else "clarification_timeout"
                        if waiting_expired
                        else "storage_limit"
                        if over_quota
                        else "worker_interrupted",
                        stopped=True,
                    )
                    del self.children[run_id]
            for row in self.store.rows(
                "SELECT r.id,s.owner FROM runs r JOIN sessions s ON s.id=r.session_id WHERE r.status='queued' ORDER BY r.created"
            ):
                if len(self.children) >= self.settings.max_workers:
                    break
                if row["id"] in self.children:
                    continue
                with self.store.transaction() as db:
                    if db.execute(
                        "SELECT 1 FROM runs r JOIN sessions s ON s.id=r.session_id WHERE s.owner=? AND r.status IN ('running','waiting_for_input','cancelling')",
                        (row["owner"],),
                    ).fetchone():
                        continue
                    # Claim before spawning, so cancellation cannot race dispatch.
                    if not db.execute(
                        "UPDATE runs SET status='running' WHERE id=? AND status='queued'",
                        (row["id"],),
                    ).rowcount:
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
                        stopped=True,
                    )
                    continue
                self.children[row["id"]] = (process, time.monotonic())
            await asyncio.sleep(0.2)
