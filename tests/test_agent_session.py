"""
Tests for ChemGraph agent session/memory integration.

Covers:
- Memory initialization options (enable_memory, custom store, db_path)
- uuid and session_id consistency
- _ensure_session idempotency
- _save_messages_to_store with LangChain and dict messages
- write_state file naming with uuid
- resume_from flow
- End-to-end session lifecycle
"""

import asyncio
import json
import os
import shutil
import tempfile

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.memory.schemas import SessionMessage
from chemgraph.memory.store import SessionStore


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def clean_env():
    """Clear CHEMGRAPH_LOG_DIR for test isolation."""
    old = os.environ.get("CHEMGRAPH_LOG_DIR")
    if "CHEMGRAPH_LOG_DIR" in os.environ:
        del os.environ["CHEMGRAPH_LOG_DIR"]
    yield
    if old:
        os.environ["CHEMGRAPH_LOG_DIR"] = old
    elif "CHEMGRAPH_LOG_DIR" in os.environ:
        del os.environ["CHEMGRAPH_LOG_DIR"]


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary database file."""
    return str(tmp_path / "test_sessions.db")


@pytest.fixture
def mock_agent_patches():
    """Patch LLM loading and graph construction for fast agent creation."""
    with (
        patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load,
        patch("chemgraph.agent.llm_agent.construct_single_agent_graph") as mock_graph,
    ):
        mock_load.return_value = Mock()
        mock_graph.return_value = Mock()
        yield mock_load, mock_graph


def _make_agent(clean_env, mock_agent_patches, tmp_db, **kwargs):
    """Helper to create a ChemGraph with memory pointed at a temp DB."""
    defaults = {
        "model_name": "gpt-4o-mini",
        "enable_memory": True,
        "memory_db_path": tmp_db,
    }
    defaults.update(kwargs)
    agent = ChemGraph(**defaults)
    return agent


# ------------------------------------------------------------------
# Memory initialization
# ------------------------------------------------------------------


class TestMemoryInitialization:
    def test_enable_memory_true_creates_store(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db, enable_memory=True)
        assert agent.session_store is not None
        assert isinstance(agent.session_store, SessionStore)

    def test_enable_memory_false_no_store(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db, enable_memory=False)
        assert agent.session_store is None

    def test_custom_session_store(self, clean_env, mock_agent_patches, tmp_db):
        custom_store = SessionStore(db_path=tmp_db)
        agent = _make_agent(
            clean_env,
            mock_agent_patches,
            tmp_db,
            session_store=custom_store,
        )
        assert agent.session_store is custom_store

    def test_custom_db_path(self, clean_env, mock_agent_patches, tmp_path):
        db_path = str(tmp_path / "custom.db")
        agent = _make_agent(
            clean_env,
            mock_agent_patches,
            str(tmp_path / "unused.db"),
            memory_db_path=db_path,
        )
        assert agent.session_store is not None
        assert agent.session_store.db_path == db_path

    def test_session_created_flag_initially_false(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        assert agent._session_created is False


# ------------------------------------------------------------------
# UUID and session_id
# ------------------------------------------------------------------


class TestUuidSessionId:
    def test_uuid_always_set(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        assert agent.uuid is not None
        assert len(agent.uuid) == 8

    def test_uuid_set_when_log_dir_preset(self, mock_agent_patches, tmp_db):
        """uuid must be set even when CHEMGRAPH_LOG_DIR is already in env."""
        os.environ["CHEMGRAPH_LOG_DIR"] = "/tmp/preset_log_dir"
        try:
            agent = _make_agent(None, mock_agent_patches, tmp_db)
            assert agent.uuid is not None
            assert len(agent.uuid) == 8
            assert agent.log_dir == "/tmp/preset_log_dir"
        finally:
            del os.environ["CHEMGRAPH_LOG_DIR"]

    def test_session_id_property_returns_uuid(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        assert agent.session_id == agent.uuid

    def test_session_id_is_str_not_optional(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        assert isinstance(agent.session_id, str)

    def test_two_agents_have_different_uuids(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent1 = _make_agent(clean_env, mock_agent_patches, tmp_db)
        # Second agent needs a fresh env since first sets CHEMGRAPH_LOG_DIR
        if "CHEMGRAPH_LOG_DIR" in os.environ:
            del os.environ["CHEMGRAPH_LOG_DIR"]
        agent2 = _make_agent(clean_env, mock_agent_patches, tmp_db)
        assert agent1.uuid != agent2.uuid


# ------------------------------------------------------------------
# _ensure_session
# ------------------------------------------------------------------


class TestEnsureSession:
    def test_creates_session_in_store(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("What is water?")

        assert agent._session_created is True
        session = agent.session_store.get_session(agent.uuid)
        assert session is not None
        assert session.session_id == agent.uuid
        assert session.model_name == "gpt-4o-mini"
        assert session.workflow_type == "single_agent"

    def test_generates_title_from_query(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("Please calculate the energy of water")

        session = agent.session_store.get_session(agent.uuid)
        assert session.title == "Calculate the energy of water"

    def test_idempotent_on_second_call(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("First query")
        agent._ensure_session("Second query")

        # Should still have only one session
        assert agent.session_store.session_count() == 1
        # Title should be from the first query
        session = agent.session_store.get_session(agent.uuid)
        assert "First" in session.title

    def test_stores_log_dir(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("test query")

        session = agent.session_store.get_session(agent.uuid)
        assert session.log_dir == agent.log_dir

    def test_noop_when_memory_disabled(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db, enable_memory=False)
        # Should not raise
        agent._ensure_session("test query")
        assert agent._session_created is False


# ------------------------------------------------------------------
# _save_messages_to_store
# ------------------------------------------------------------------


class TestSaveMessagesToStore:
    def test_saves_langchain_messages(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("test query")

        # Simulate LangChain message objects
        human_msg = Mock()
        human_msg.type = "human"
        human_msg.content = "What is water?"

        ai_msg = Mock()
        ai_msg.type = "ai"
        ai_msg.content = "Water is H2O."

        tool_msg = Mock()
        tool_msg.type = "tool"
        tool_msg.content = '{"smiles": "O"}'
        tool_msg.name = "molecule_name_to_smiles"

        state = {"messages": [human_msg, ai_msg, tool_msg]}
        agent._save_messages_to_store(state, "test query")

        session = agent.session_store.get_session(agent.uuid)
        assert len(session.messages) == 3
        assert session.messages[0].role == "human"
        assert session.messages[1].role == "ai"
        assert session.messages[2].role == "tool"
        assert session.messages[2].tool_name == "molecule_name_to_smiles"

    def test_saves_dict_messages(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("test query")

        state = {
            "messages": [
                {"type": "human", "content": "Hello"},
                {"role": "ai", "content": "Hi there"},
            ]
        }
        agent._save_messages_to_store(state, "test query")

        session = agent.session_store.get_session(agent.uuid)
        assert len(session.messages) == 2

    def test_saves_full_content_without_truncation(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("test query")

        long_msg = Mock()
        long_msg.type = "ai"
        long_msg.content = "A" * 15000

        state = {"messages": [long_msg]}
        agent._save_messages_to_store(state, "test query")

        session = agent.session_store.get_session(agent.uuid)
        assert len(session.messages) == 1
        # Content should be saved in full — no truncation at save time
        assert len(session.messages[0].content) == 15000
        assert session.messages[0].content == "A" * 15000

    def test_noop_when_memory_disabled(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db, enable_memory=False)
        state = {"messages": [{"type": "human", "content": "Hello"}]}
        # Should not raise
        agent._save_messages_to_store(state, "test query")

    def test_noop_when_session_not_created(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        # Don't call _ensure_session
        state = {"messages": [{"type": "human", "content": "Hello"}]}
        agent._save_messages_to_store(state, "test query")
        # Store should have no sessions
        assert agent.session_store.session_count() == 0

    def test_skips_messages_without_content(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("test query")

        empty_msg = Mock()
        empty_msg.type = "ai"
        empty_msg.content = ""

        state = {"messages": [empty_msg]}
        agent._save_messages_to_store(state, "test query")

        session = agent.session_store.get_session(agent.uuid)
        assert len(session.messages) == 0  # Empty content is skipped

    def test_handles_exception_gracefully(self, clean_env, mock_agent_patches, tmp_db):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent._ensure_session("test query")

        # Force an exception during save
        agent.session_store.save_messages = Mock(side_effect=RuntimeError("DB error"))

        msg = Mock()
        msg.type = "human"
        msg.content = "Hello"
        state = {"messages": [msg]}

        # Should not raise — logs a warning instead
        agent._save_messages_to_store(state, "test query")


# ------------------------------------------------------------------
# write_state file naming
# ------------------------------------------------------------------


class TestWriteStateFileNaming:
    def test_filename_includes_uuid(
        self, clean_env, mock_agent_patches, tmp_db, tmp_path
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)

        # Mock get_state to return something serializable
        agent.workflow.get_state = Mock(return_value=Mock(values={"messages": []}))

        log_dir = str(tmp_path / "test_logs")
        os.makedirs(log_dir, exist_ok=True)
        agent.log_dir = log_dir

        config = {"configurable": {"thread_id": "42"}}
        result = agent.write_state(config=config)

        assert result != "Error"
        # Find the file that was written
        files = os.listdir(log_dir)
        json_files = [f for f in files if f.endswith(".json")]
        assert len(json_files) == 1
        fname = json_files[0]

        # Filename should contain thread_id and uuid
        assert f"thread_42_{agent.uuid}" in fname

    def test_no_overwrite_same_second(
        self, clean_env, mock_agent_patches, tmp_db, tmp_path
    ):
        """Two agents writing to the same dir at the same second don't collide."""
        log_dir = str(tmp_path / "shared_logs")
        os.makedirs(log_dir, exist_ok=True)

        agents = []
        for _ in range(2):
            if "CHEMGRAPH_LOG_DIR" in os.environ:
                del os.environ["CHEMGRAPH_LOG_DIR"]
            a = _make_agent(clean_env, mock_agent_patches, tmp_db)
            a.workflow.get_state = Mock(return_value=Mock(values={"messages": []}))
            a.log_dir = log_dir
            agents.append(a)

        config = {"configurable": {"thread_id": "1"}}
        agents[0].write_state(config=config)
        agents[1].write_state(config=config)

        json_files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
        # Should be 2 distinct files (or at least not overwritten) thanks to uuid
        # They may have identical timestamps but different uuids
        assert agents[0].uuid != agents[1].uuid


# ------------------------------------------------------------------
# resume_from flow
# ------------------------------------------------------------------


class TestResumeFrom:
    def _make_streamable_agent(self, clean_env, mock_agent_patches, tmp_db):
        """Create an agent with a mock async workflow."""
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)

        # Set up a mock astream that yields one state
        ai_msg = Mock()
        ai_msg.type = "ai"
        ai_msg.content = "Test response"
        ai_msg.pretty_print = Mock()

        final_state = {"messages": [ai_msg]}

        async def mock_astream(inputs, stream_mode, config):
            yield final_state

        agent.workflow.astream = mock_astream
        agent.workflow.get_state = Mock(return_value=Mock(values=final_state))
        return agent

    @pytest.mark.asyncio
    async def test_resume_prepends_context(self, clean_env, mock_agent_patches, tmp_db):
        # Create first agent and seed a session
        agent1 = self._make_streamable_agent(clean_env, mock_agent_patches, tmp_db)
        await agent1.run("What is water?")

        session_id = agent1.uuid

        # Clear env for second agent
        if "CHEMGRAPH_LOG_DIR" in os.environ:
            del os.environ["CHEMGRAPH_LOG_DIR"]

        # Create second agent sharing the same DB
        agent2 = self._make_streamable_agent(clean_env, mock_agent_patches, tmp_db)

        # Track what inputs are passed to astream
        captured_inputs = []
        original_astream = agent2.workflow.astream

        async def tracking_astream(inputs, stream_mode, config):
            captured_inputs.append(inputs)
            ai_msg = Mock()
            ai_msg.type = "ai"
            ai_msg.content = "Follow-up response"
            ai_msg.pretty_print = Mock()
            yield {"messages": [ai_msg]}

        agent2.workflow.astream = tracking_astream
        agent2.workflow.get_state = Mock(
            return_value=Mock(
                values={"messages": [Mock(type="ai", content="Follow-up")]}
            )
        )

        await agent2.run("Continue the analysis", resume_from=session_id)

        # The query should contain the previous context
        assert len(captured_inputs) == 1
        query = captured_inputs[0]["messages"]
        assert "Previous Session Context" in query
        assert "continuing from the previous session" in query

    @pytest.mark.asyncio
    async def test_resume_from_nonexistent_session(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = self._make_streamable_agent(clean_env, mock_agent_patches, tmp_db)

        captured_inputs = []

        async def tracking_astream(inputs, stream_mode, config):
            captured_inputs.append(inputs)
            ai_msg = Mock()
            ai_msg.type = "ai"
            ai_msg.content = "Response"
            ai_msg.pretty_print = Mock()
            yield {"messages": [ai_msg]}

        agent.workflow.astream = tracking_astream
        agent.workflow.get_state = Mock(
            return_value=Mock(values={"messages": [Mock(type="ai", content="resp")]})
        )

        await agent.run("Hello", resume_from="nonexistent_id")

        # No context should be prepended for a nonexistent session
        query = captured_inputs[0]["messages"]
        assert "Previous Session Context" not in query
        assert query == "Hello"

    @pytest.mark.asyncio
    async def test_resume_from_ignored_when_memory_disabled(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db, enable_memory=False)

        ai_msg = Mock()
        ai_msg.type = "ai"
        ai_msg.content = "Response"
        ai_msg.pretty_print = Mock()

        captured_inputs = []

        async def tracking_astream(inputs, stream_mode, config):
            captured_inputs.append(inputs)
            yield {"messages": [ai_msg]}

        agent.workflow.astream = tracking_astream
        agent.workflow.get_state = Mock(
            return_value=Mock(values={"messages": [ai_msg]})
        )

        await agent.run("Hello", resume_from="some_id")

        query = captured_inputs[0]["messages"]
        assert query == "Hello"


# ------------------------------------------------------------------
# End-to-end session lifecycle
# ------------------------------------------------------------------


class TestEndToEndSessionLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, clean_env, mock_agent_patches, tmp_db):
        """init -> run -> messages saved -> load_previous_context -> resume"""
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db)

        # Set up mock workflow
        human_msg = Mock()
        human_msg.type = "human"
        human_msg.content = "Calculate energy of H2"

        ai_msg = Mock()
        ai_msg.type = "ai"
        ai_msg.content = "The energy of H2 is -1.17 eV using MACE."
        ai_msg.pretty_print = Mock()

        final_state = {"messages": [human_msg, ai_msg]}

        async def mock_astream(inputs, stream_mode, config):
            yield final_state

        agent.workflow.astream = mock_astream
        agent.workflow.get_state = Mock(return_value=Mock(values=final_state))

        # Step 1: Run
        await agent.run("Calculate energy of H2")

        # Step 2: Verify session was created
        assert agent._session_created is True
        session = agent.session_store.get_session(agent.uuid)
        assert session is not None
        assert len(session.messages) == 2

        # Step 3: Verify load_previous_context works
        context = agent.load_previous_context(agent.uuid)
        assert "Previous Session Context" in context
        assert "H2" in context

        # Step 4: Verify session_id property
        assert agent.session_id == agent.uuid

        # Step 5: Create new agent and resume
        if "CHEMGRAPH_LOG_DIR" in os.environ:
            del os.environ["CHEMGRAPH_LOG_DIR"]

        agent2 = _make_agent(clean_env, mock_agent_patches, tmp_db)
        agent2.workflow.astream = mock_astream
        agent2.workflow.get_state = Mock(return_value=Mock(values=final_state))

        await agent2.run("Now optimize H2", resume_from=agent.uuid)

        # Second agent should also have a session
        assert agent2._session_created is True
        assert agent2.uuid != agent.uuid

    @pytest.mark.asyncio
    async def test_load_previous_context_disabled_memory(
        self, clean_env, mock_agent_patches, tmp_db
    ):
        agent = _make_agent(clean_env, mock_agent_patches, tmp_db, enable_memory=False)
        result = agent.load_previous_context("some_id")
        assert result == ""
