"""Scientific runners with explicit restart behavior."""

from dataclasses import asdict, dataclass, field


@dataclass
class Outcome:
    state: str = "completed"
    text: str = ""
    reason: str = ""
    interrupts: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)


class CheckpointReviewRequired(RuntimeError):
    """A saved scientific thread cannot safely use the current runtime."""


class ChemGraphAdapter:
    contract_version = 2

    def __init__(self, settings):
        self.settings = settings
        self.sessions = {}
        self.contexts = {}

    async def start(self):
        from chemgraph.execution.scoped import EXECUTION_CONTRACT_VERSION
        from chemgraph.memory.store import SessionStore
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        if EXECUTION_CONTRACT_VERSION != self.contract_version:
            raise RuntimeError("ChemGraph service requires execution contract version 2")

        self.saver_context = AsyncSqliteSaver.from_conn_string(
            str(self.settings.state_dir / "checkpoints.db")
        )
        self.saver = await self.saver_context.__aenter__()
        await self.saver.setup()
        self.session_store = SessionStore(str(self.settings.state_dir / "sessions.db"))

    async def close(self):
        if hasattr(self, "saver_context"):
            await self.saver_context.__aexit__(None, None, None)

    async def session(self, record):
        from chemgraph.agent.llm_agent import ChemGraph
        from chemgraph.agent.main_session import MainAgentRestoreError, MainAgentSession
        from chemgraph.execution.submissions import submission_tool
        from chemgraph.tools.generic_tools import ask_human

        thread = record["thread_id"]
        if thread in self.sessions:
            return self.sessions[thread]
        directory = self.settings.state_dir / "artifacts" / thread
        directory.mkdir(parents=True, exist_ok=True)

        selected_tools = [ask_human]
        from chemgraph.registry.tools import ToolRegistry, BUILTIN_TOOL_SPECS

        registry = ToolRegistry(
            spec for spec in BUILTIN_TOOL_SPECS if spec.name in self.settings.profile.capabilities
        )
        if self.settings.profile.hpc:
            selected_tools.append(submission_tool(self))
        from chemgraph.execution.scoped import execution_context

        with execution_context(self.context(record)):
            agent = ChemGraph(
                workflow_type="main_agent",
                model_name=self.settings.profile.model,
                tools=selected_tools,
                data_tools=[],
                human_supervised=True,
                approval_mode="review",
                tool_registry=registry,
                subagent_names=[],
                discover_skills=False,
                deepagent_discover_skills=False,
                skills=[],
                skill_dirs=[],
                checkpointer=self.saver,
                session_store=self.session_store,
                log_dir=str(directory),
                recursion_limit=self.settings.profile.recursion_limit,
                configuration_id=self.settings.profile.identity,
            )
        session = MainAgentSession(
            agent.workflow,
            thread_id=thread,
            session_store=self.session_store,
            session_metadata=agent.main_agent_metadata,
        )
        snapshot = await session.workflow.aget_state(session.config)
        if snapshot.created_at:
            try:
                await session.restore()
            except MainAgentRestoreError as exc:
                raise CheckpointReviewRequired(
                    "Checkpoint configuration must be reconciled"
                ) from exc
        elif self.session_store.get_session_metadata(thread) is not None:
            raise CheckpointReviewRequired("Readable session exists but its checkpoint is missing")
        self.sessions[thread] = session
        return session

    async def prepare(self, record):
        session = await self.session(record)
        snapshot = await session.workflow.aget_state(session.config)
        return snapshot.config if snapshot.created_at else None

    def context(self, record):
        from chemgraph.execution.scoped import ExecutionContext

        thread = record["thread_id"]
        if thread not in self.contexts:
            self.contexts[thread] = ExecutionContext(
                owner=record["owner"],
                thread_id=thread,
                task_id=record["id"],
                workspace=self.settings.state_dir / "artifacts" / thread,
                controller=self,
                model_roots=self.settings.profile.model_roots,
                policy=self.settings.profile.execution_policy(),
            )
        context = self.contexts[thread]
        if context.owner != record["owner"]:
            raise ValueError("Scientific context belongs to another caller")
        context.task_id = record["id"]
        return context

    async def execute_tool(self, context, call, stages):
        return await self.operations.execute_tool(self.current_record, context, call, stages)

    async def prepare_submission(self, call_id, calculation):
        return await self.operations.prepare(self.current_record, call_id, calculation)

    async def execute_submission(self, identifier):
        return await self.operations.execute(identifier)

    async def run(self, record):
        from chemgraph.execution.scoped import execution_context

        self.current_record = record
        session = await self.session(record)
        with execution_context(self.context(record)):
            if record["action"] == "recover":
                await session.restore()
                result = await session.retry()
            elif record["action"] == "resume":
                responses = await self.operations.resume_responses(record)
                if responses is None:
                    return Outcome(
                        state="input_required",
                        reason="approval",
                        text="Operator authorization expired; review the pending actions again.",
                        interrupts=record["interrupts"],
                    )
                result = await session.resume(responses)
            else:
                result = await session.run(record["text"])
        return await self.outcome(record, result)

    async def outcome(self, record, result):
        from chemgraph.execution.scoped import result_artifacts

        pending = [asdict(item) for item in result.interrupts]
        await self.operations.register_reviews(record, pending, result.state)
        # Tool reviews are never satisfied by an ordinary peer message.
        approval = any(
            isinstance(i["payload"], dict)
            and ("action_requests" in i["payload"] or i["payload"].get("kind") == "approval")
            for i in pending
        )
        reconciliation = any(
            isinstance(i["payload"], dict) and i["payload"].get("kind") == "reconciliation"
            for i in pending
        )
        state = {
            "completed": "completed",
            "waiting_for_user": "input_required",
            "failed": "failed",
        }[result.status]
        artifacts = result_artifacts(result.state, record["id"])
        return Outcome(
            state=state,
            text=result.assistant_response or "Input required",
            reason=(
                "reconciliation" if reconciliation else "approval" if approval else "clarification"
            )
            if pending
            else "",
            interrupts=pending,
            artifacts=artifacts,
        )

    async def recover(self, record):
        session = await self.session(record)
        snapshot = await session.workflow.aget_state(session.config)
        # A completed previous turn is not evidence that this accepted message ran.
        if snapshot.created_at and snapshot.config != record.get("checkpoint_before"):
            result = await session.restore()
            if result.status != "failed":
                return await self.outcome(record, result)
        return Outcome(
            state="input_required",
            reason="reconciliation",
            text="Execution was interrupted. Inspect checkpoints and operations before retrying.",
        )
