"""Unattended execution of durable main-agent sessions."""

from rich.markup import escape

from chemgraph.agent.configuration import validate_cli_configuration
from chemgraph.agent.main_session import IncompatibleCheckpointError
from chemgraph.cli import commands
from chemgraph.cli.checkpoint_runtime import CheckpointRuntime, DEFAULT_CHECKPOINT_DB
from chemgraph.cli.formatting import console, format_response
from chemgraph.memory.store import SessionStore
from chemgraph.registry.tools import RegistryError


def run_headless_main_agent(*, query=None, resume_session=None, checkpoint_db=None,
                            output_file=None, **options) -> int:
    """Run or inspect a thread without prompting or automatically retrying it."""
    runtime = None
    session = None
    try:
        if options.get("approval_mode") != "bypass":
            raise ValueError("Headless main_agent requires --dangerously-skip-approvals.")
        if not query and not resume_session:
            raise ValueError("Query is required unless --resume specifies a saved session.")
        if query is not None and not query.strip():
            raise ValueError("Query must not be empty.")
        saved = None
        if resume_session:
            saved = SessionStore().get_session_metadata(resume_session)
            if saved is None or saved[1] is None:
                raise ValueError("No durable main-agent session matches --resume.")
            thread_id, metadata = saved
            if metadata.checkpoint_backend != "AsyncSqliteSaver":
                raise ValueError("This session does not use a CLI-restorable SQLite checkpoint.")
            validate_cli_configuration(metadata.graph_config)
            commands.validate_main_agent_approval(metadata.graph_config, options["approval_mode"])
            checkpoint_db = metadata.checkpoint_db or checkpoint_db
            commands._print_main_agent_restore_configuration(thread_id, metadata.graph_config)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 2
    except Exception as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 1

    try:
        runtime = CheckpointRuntime()
        checkpoint_db = checkpoint_db or DEFAULT_CHECKPOINT_DB
        saver = runtime.open_sqlite(checkpoint_db)
        if saved:
            agent = commands._initialize_saved_main_agent(
                metadata.graph_config, return_option=options["return_option"],
                checkpointer=saver, argo_user=options.get("argo_user"),
                verbose=options.get("verbose", False), approval_mode=options["approval_mode"],
                _raise_configuration_errors=True,
            )
        else:
            agent = commands.initialize_agent(
                workflow_type="main_agent", checkpointer=saver, _raise_configuration_errors=True, **options,
            )
        if agent is None:
            return 1
        session = commands.create_main_agent_session(
            agent, thread_id=thread_id if saved else None, checkpoint_db=checkpoint_db,
        )
        result = None
        if saved:
            result = commands.restore_main_agent_session(
                session, checkpoint_runtime=runtime, interactive=False,
            )
        if query and (not saved or (result is not None and result.status == "completed")):
            result = commands.run_main_agent_query(
                session, query, verbose=options.get("verbose", False),
                checkpoint_runtime=runtime, interactive=False,
            )
        status = result.status if result is not None else "failed"
        console.print(f"Session: {escape(session.thread_id)} | Status: {status}")
        if result is not None:
            format_response(result, verbose=options.get("verbose", False))
            if status == "waiting_for_user":
                for pending in result.interrupts:
                    console.print(escape(commands._interrupt_question(pending.payload)))
                console.print("Resume interactively with --resume and --dangerously-skip-approvals to answer.")
            elif status == "failed":
                console.print("Resume interactively with --resume and --dangerously-skip-approvals; use /retry.")
            if output_file:
                commands.save_output(str(result), output_file)
        commands.print_token_usage(session, stderr=True)
        return {"completed": 0, "waiting_for_user": 3}.get(status, 1)
    except (IncompatibleCheckpointError, TypeError, ValueError, RegistryError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 2
    except KeyboardInterrupt:
        if session is not None:
            console.print(f"Session: {escape(session.thread_id)} | Interrupted")
        return 130
    except Exception as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 1
    finally:
        if runtime is not None:
            runtime.close()
