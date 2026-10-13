"""Unattended execution of durable main-agent sessions."""

import json
import os
from pathlib import Path
import tempfile

from rich.console import Console
from rich.markup import escape

from chemgraph.agent.configuration import validate_cli_configuration
from chemgraph.agent.main_session import IncompatibleCheckpointError
from chemgraph.agent.interrupts import is_tool_review
from chemgraph.cli import commands
from chemgraph.cli.checkpoint_runtime import CheckpointRuntime, DEFAULT_CHECKPOINT_DB
from chemgraph.cli.formatting import console, format_response
from chemgraph.memory.graph_config import describe_tool_registry
from chemgraph.memory.store import SessionStore
from chemgraph.registry.tools import RegistryError


HEADLESS_TOOLS_ERROR = (
    "Headless main_agent cannot restore MCP or attached tools. Use --interactive; "
    "these sessions do not support CLI restoration."
)
HEADLESS_CONFIGURATION_ERROR = (
    "Headless main_agent requires a CLI-restorable configuration. "
    "Use --interactive for this configuration; CLI restoration is unavailable."
)


class _HandoffError(ValueError):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


def _interrupt_kind(payload):
    if is_tool_review(payload):
        return "tool_review"
    if (isinstance(payload, dict) and set(payload) == {"question"}
            and isinstance(payload["question"], str) and payload["question"].strip()):
        return "clarification"
    return "unsupported"


def _receipt_interrupt(item):
    payload, kind = item.payload, _interrupt_kind(item.payload)
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        payload = {"message": "Interrupt payload could not be encoded as JSON; inspect interactively."}
        kind = "unsupported"
    return {"id": item.id, "kind": kind, "payload": payload}


def _load_answers(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate answer ID: {key}")
            result[key] = value
        return result

    try:
        answers = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    except (OSError, ValueError) as exc:
        raise _HandoffError(f"Invalid --answers-file: {exc}", "invalid_answers") from exc
    if not isinstance(answers, dict) or not answers or any(
        not key or not isinstance(value, str) or not value.strip()
        for key, value in answers.items()
    ):
        raise _HandoffError("--answers-file must map interrupt IDs to nonempty answer strings.", "invalid_answers")
    return answers


def _validate_answers(result, answers):
    if result is None or result.status != "waiting_for_user" or not result.interrupts:
        raise _HandoffError("Answers require a session waiting for clarification.", "not_waiting")
    if any(not item.id for item in result.interrupts):
        raise _HandoffError("Pending questions must have stable interrupt IDs.", "unsupported_interrupt")
    # The session API also accepts review payloads; enforce IDs even for one question.
    if set(answers) != {item.id for item in result.interrupts}:
        raise _HandoffError("Answer IDs must exactly match the current pending interrupts.", "stale_answers")
    if any(_interrupt_kind(item.payload) != "clarification" for item in result.interrupts):
        raise _HandoffError(
            "--answers-file supports clarifications only; resume interactively for other interrupts.",
            "unsupported_interrupt",
        )


def _validate_handoff_paths(*paths):
    paths = [Path(path).resolve() for path in paths if path is not None]
    for index, path in enumerate(paths):
        for other in paths[:index]:
            if path == other or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError("Answer, result, and output files must have distinct paths.")


def _write_result_file(path, session, result, error, *, exit_code, error_code=None, status=None):
    receipt = {
        "version": 1,
        "session_id": session.thread_id if session is not None else None,
        "status": status or (result.status if result is not None else (
            "failed" if session is not None and session.failed else None
        )),
        "assistant_response": result.assistant_response if result is not None else "",
        "interrupts": [_receipt_interrupt(item) for item in result.interrupts] if result is not None else [],
        "error": error,
        "error_code": error_code,
        "exit_code": exit_code,
    }
    content = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def report_headless_result(code, *, result_file=None, session=None, result=None,
                           error=None, error_code=None, status=None):
    """Write a receipt when requested; None denotes an invocation still running."""
    if result_file is not None:
        if error is None and code == 1:
            error = "Execution failed; inspect the CLI log and saved session before retrying."
        error_code = error_code or {1: "execution", 2: "config", 130: "interrupted"}.get(code)
        try:
            _write_result_file(result_file, session, result, error,
                               exit_code=code, error_code=error_code, status=status)
        except (OSError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            console.print(f"[red]Error writing result file: {escape(str(exc))}[/red]")
            return 1
    return code


def run_headless_main_agent(*, query=None, resume_session=None, checkpoint_db=None,
                            output_file=None, result_file=None, answers_file=None, **options) -> int:
    """Run or inspect a thread without prompting or automatically retrying it."""
    runtime = None
    session = None
    result = None
    receipt_path = None

    def finish(code, error=None, error_code=None, *, status=None):
        return report_headless_result(code, result_file=receipt_path, session=session, result=result,
                                      error=error, error_code=error_code, status=status)

    try:
        if result_file is not None or answers_file is not None:
            _validate_handoff_paths(answers_file, result_file, output_file)
            receipt_path = result_file
        if result_file is not None and not Path(result_file).parent.is_dir():
            raise OSError("The --result-file parent directory must already exist.")
        if answers_file is not None and (not resume_session or query is not None):
            raise ValueError("--answers-file requires --resume and cannot be combined with --query.")
        answers = _load_answers(answers_file) if answers_file is not None else None
        if options.get("approval_mode") != "bypass":
            raise ValueError("Headless main_agent requires --dangerously-skip-approvals.")
        if not query and not resume_session:
            raise ValueError("Query is required unless --resume specifies a saved session.")
        if query is not None and not query.strip():
            raise ValueError("Query must not be empty.")
        if not resume_session:
            if options.get("tools"):
                raise ValueError(HEADLESS_TOOLS_ERROR)
            registry = options.get("tool_registry")
            if registry is not None and describe_tool_registry(registry)[1]:
                raise ValueError(HEADLESS_CONFIGURATION_ERROR)
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
    except _HandoffError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return finish(2, str(exc), exc.error_code)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return finish(2, str(exc))
    except Exception as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return finish(1, str(exc))

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
            return finish(1, "Could not initialize the main agent.")
        if not saved:
            try:
                validate_cli_configuration(agent.main_agent_metadata.graph_config)
            except ValueError as exc:
                raise ValueError(HEADLESS_CONFIGURATION_ERROR) from exc
        session = commands.create_main_agent_session(
            agent, thread_id=thread_id if saved else None, checkpoint_db=checkpoint_db,
        )
        if saved:
            result = commands.restore_main_agent_session(
                session, checkpoint_runtime=runtime, interactive=False,
            )
            if query and result is not None and result.status in {"waiting_for_user", "failed"}:
                Console(stderr=True).print(f"Query not run: session is {result.status}.")
        if answers is not None:
            if result is None:
                return finish(1, "Could not restore the session; answers were not submitted.")
            _validate_answers(result, answers)
            # Clear restored state so a failed operation cannot report the previous result.
            result = None
            if finish(None, status="running") == 1:
                return 1
            result = commands._run_main_agent_operation(
                session, lambda: session.resume(answers),
                progress_description="Answering main-agent questions...",
                checkpoint_runtime=runtime, interactive=False, report_failure_usage=False,
            )
        if query and (not saved or (result is not None and result.status == "completed")):
            # Clear restored state so a failed operation cannot report the previous result.
            result = None
            if finish(None, status="running") == 1:
                return 1
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
                if result_file and all(_interrupt_kind(item.payload) == "clarification" for item in result.interrupts):
                    console.print("Or submit answers by interrupt ID with --resume and --answers-file.")
            elif status == "failed":
                console.print("Resume interactively with --resume and --dangerously-skip-approvals; use /retry.")
            if output_file:
                try:
                    content = (result.assistant_response if options["return_option"] == "last_message"
                               else json.dumps(result.state, ensure_ascii=False, indent=2, allow_nan=False))
                except (TypeError, ValueError) as exc:
                    console.print(f"[red]Error serializing output: {escape(str(exc))}[/red]")
                    return finish(1, f"Error serializing output: {exc}", "export")
                if not commands.save_output(content, output_file):
                    return finish(1, "Could not write --output-file; the saved session is unchanged.", "export")
        return finish({"completed": 0, "waiting_for_user": 3}.get(status, 1))
    except _HandoffError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return finish(2, str(exc), exc.error_code)
    except (IncompatibleCheckpointError, TypeError, ValueError, RegistryError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return finish(2, str(exc))
    except KeyboardInterrupt:
        if session is not None:
            console.print(f"Session: {escape(session.thread_id)} | Interrupted")
        return finish(130, "Interrupted; inspect the saved session before continuing.")
    except Exception as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return finish(1, str(exc))
    finally:
        if runtime is not None:
            runtime.close()
        if session is not None:
            commands.print_token_usage(session, stderr=True)
