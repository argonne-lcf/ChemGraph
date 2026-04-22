"""
Tests for ChemGraph session memory storage.
"""

import os
from datetime import datetime

import pytest

from chemgraph.memory.schemas import Session, SessionMessage, SessionSummary
from chemgraph.memory.store import SessionStore


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database file for testing."""
    return str(tmp_path / "test_sessions.db")


@pytest.fixture
def store(tmp_db):
    """Create a SessionStore with a temporary database."""
    return SessionStore(db_path=tmp_db)


# ------------------------------------------------------------------
# Schema tests
# ------------------------------------------------------------------


class TestSchemas:
    def test_session_message_creation(self):
        msg = SessionMessage(role="human", content="Hello world")
        assert msg.role == "human"
        assert msg.content == "Hello world"
        assert msg.tool_name is None
        assert isinstance(msg.timestamp, datetime)

    def test_session_message_tool(self):
        msg = SessionMessage(role="tool", content="Result: 42", tool_name="calculator")
        assert msg.role == "tool"
        assert msg.tool_name == "calculator"

    def test_session_creation(self):
        session = Session(
            session_id="abc12345",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        assert session.session_id == "abc12345"
        assert session.title == ""
        assert session.messages == []
        assert session.query_count == 0

    def test_session_summary(self):
        summary = SessionSummary(
            session_id="abc12345",
            title="Test session",
            model_name="gpt-4o",
            workflow_type="single_agent",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            query_count=3,
            message_count=10,
        )
        assert summary.query_count == 3
        assert summary.message_count == 10


# ------------------------------------------------------------------
# Store tests
# ------------------------------------------------------------------


class TestSessionStore:
    def test_init_creates_db(self, tmp_db):
        assert os.path.exists(tmp_db)

    def test_create_session(self, store):
        session = store.create_session(
            session_id="test1234",
            model_name="gpt-4o-mini",
            workflow_type="single_agent",
            title="Test Session",
        )
        assert session.session_id == "test1234"
        assert session.title == "Test Session"
        assert session.model_name == "gpt-4o-mini"

    def test_get_session(self, store):
        store.create_session(
            session_id="test1234",
            model_name="gpt-4o-mini",
            workflow_type="single_agent",
            title="Test Session",
        )

        session = store.get_session("test1234")
        assert session is not None
        assert session.session_id == "test1234"
        assert session.title == "Test Session"

    def test_get_session_not_found(self, store):
        session = store.get_session("nonexistent")
        assert session is None

    def test_save_and_retrieve_messages(self, store):
        store.create_session(
            session_id="msg_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        messages = [
            SessionMessage(role="human", content="What is water?"),
            SessionMessage(role="ai", content="Water is H2O."),
            SessionMessage(
                role="tool",
                content='{"smiles": "O"}',
                tool_name="molecule_name_to_smiles",
            ),
        ]

        store.save_messages("msg_test", messages)

        session = store.get_session("msg_test")
        assert session is not None
        assert len(session.messages) == 3
        assert session.messages[0].role == "human"
        assert session.messages[0].content == "What is water?"
        assert session.messages[1].role == "ai"
        assert session.messages[2].tool_name == "molecule_name_to_smiles"

    def test_save_messages_updates_query_count(self, store):
        store.create_session(
            session_id="count_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        messages = [
            SessionMessage(role="human", content="Query 1"),
            SessionMessage(role="ai", content="Response 1"),
            SessionMessage(role="human", content="Query 2"),
            SessionMessage(role="ai", content="Response 2"),
        ]

        store.save_messages("count_test", messages)

        session = store.get_session("count_test")
        assert session.query_count == 2  # Only counts human messages

    def test_save_messages_updates_title(self, store):
        store.create_session(
            session_id="title_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        messages = [SessionMessage(role="human", content="Hello")]
        store.save_messages("title_test", messages, title="New Title")

        session = store.get_session("title_test")
        assert session.title == "New Title"

    def test_list_sessions(self, store):
        for i in range(5):
            store.create_session(
                session_id=f"list_{i}",
                model_name="gpt-4o",
                workflow_type="single_agent",
                title=f"Session {i}",
            )

        sessions = store.list_sessions()
        assert len(sessions) == 5
        # Should be ordered by updated_at DESC
        for s in sessions:
            assert isinstance(s, SessionSummary)

    def test_list_sessions_with_limit(self, store):
        for i in range(10):
            store.create_session(
                session_id=f"limit_{i}",
                model_name="gpt-4o",
                workflow_type="single_agent",
            )

        sessions = store.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_with_offset(self, store):
        for i in range(5):
            store.create_session(
                session_id=f"offset_{i}",
                model_name="gpt-4o",
                workflow_type="single_agent",
            )

        offset_sessions = store.list_sessions(offset=2)
        assert len(offset_sessions) == 3

    def test_delete_session(self, store):
        store.create_session(
            session_id="del_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        # Add some messages
        messages = [
            SessionMessage(role="human", content="Hello"),
            SessionMessage(role="ai", content="Hi!"),
        ]
        store.save_messages("del_test", messages)

        assert store.delete_session("del_test") is True
        assert store.get_session("del_test") is None

    def test_delete_session_not_found(self, store):
        assert store.delete_session("nonexistent") is False

    def test_session_count(self, store):
        assert store.session_count() == 0

        store.create_session(
            session_id="count1",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        assert store.session_count() == 1

        store.create_session(
            session_id="count2",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        assert store.session_count() == 2

    def test_prefix_resolution(self, store):
        store.create_session(
            session_id="abcd1234",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        # Exact match
        session = store.get_session("abcd1234")
        assert session is not None

        # Prefix match
        session = store.get_session("abcd")
        assert session is not None
        assert session.session_id == "abcd1234"

    def test_ambiguous_prefix(self, store):
        store.create_session(
            session_id="abc_one",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        store.create_session(
            session_id="abc_two",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        # "abc" matches both - should return None
        session = store.get_session("abc")
        assert session is None


# ------------------------------------------------------------------
# Context building tests
# ------------------------------------------------------------------


class TestContextBuilding:
    def test_build_context_messages(self, store):
        store.create_session(
            session_id="ctx_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        messages = [
            SessionMessage(role="human", content="What is water?"),
            SessionMessage(role="ai", content="Water is H2O."),
            SessionMessage(role="tool", content="tool output", tool_name="lookup"),
            SessionMessage(role="human", content="What about ethanol?"),
            SessionMessage(role="ai", content="Ethanol is C2H5OH."),
        ]
        store.save_messages("ctx_test", messages)

        # Default: human + ai + tool
        ctx = store.build_context_messages("ctx_test")
        assert len(ctx) == 5  # 2 human + 2 ai + 1 tool
        assert all(m["role"] in ("human", "ai", "tool") for m in ctx)

    def test_build_context_messages_with_limit(self, store):
        store.create_session(
            session_id="ctx_limit",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        messages = [
            SessionMessage(role="human", content=f"Query {i}") for i in range(10)
        ]
        store.save_messages("ctx_limit", messages)

        ctx = store.build_context_messages("ctx_limit", max_messages=3)
        assert len(ctx) == 3
        # Should be the last 3
        assert ctx[0]["content"] == "Query 7"

    def test_build_context_messages_not_found(self, store):
        ctx = store.build_context_messages("nonexistent")
        assert ctx == []

    def test_build_context_summary(self, store):
        store.create_session(
            session_id="sum_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
            title="Water Analysis",
        )
        messages = [
            SessionMessage(role="human", content="What is water?"),
            SessionMessage(role="tool", content='{"smiles": "O"}', tool_name="lookup"),
            SessionMessage(role="ai", content="Water is H2O, a simple molecule."),
        ]
        store.save_messages("sum_test", messages)

        summary = store.build_context_summary("sum_test")
        assert "Previous Session Context" in summary
        assert "Water Analysis" in summary
        assert "What is water?" in summary
        assert "Water is H2O" in summary
        assert "Tool [lookup]" in summary
        assert '{"smiles": "O"}' in summary

    def test_build_context_summary_not_found(self, store):
        summary = store.build_context_summary("nonexistent")
        assert summary == ""

    def test_build_context_summary_truncates_long_ai(self, store):
        store.create_session(
            session_id="trunc_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        long_response = "A" * 1000
        messages = [
            SessionMessage(role="human", content="Give me a long answer"),
            SessionMessage(role="ai", content=long_response),
        ]
        store.save_messages("trunc_test", messages)

        summary = store.build_context_summary("trunc_test")
        assert "..." in summary


# ------------------------------------------------------------------
# Title generation tests
# ------------------------------------------------------------------


class TestTitleGeneration:
    def test_generate_title_basic(self):
        title = SessionStore.generate_title("What is the energy of water?")
        assert title == "What is the energy of water?"

    def test_generate_title_strips_prefix(self):
        title = SessionStore.generate_title("Please calculate the energy of water")
        assert title == "Calculate the energy of water"

    def test_generate_title_truncates(self):
        long_query = "A" * 100
        title = SessionStore.generate_title(long_query, max_length=20)
        assert len(title) <= 20
        assert title.endswith("...")

    def test_generate_title_capitalizes(self):
        title = SessionStore.generate_title("calculate energy")
        assert title[0] == "C"

    def test_generate_title_empty(self):
        title = SessionStore.generate_title("")
        assert title == ""


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_messages_save(self, store):
        store.create_session(
            session_id="empty_msg",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        # Should not raise
        store.save_messages("empty_msg", [])

        session = store.get_session("empty_msg")
        assert len(session.messages) == 0

    def test_multiple_message_batches(self, store):
        store.create_session(
            session_id="batch_test",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        # First batch
        store.save_messages(
            "batch_test",
            [SessionMessage(role="human", content="First query")],
        )

        # Second batch
        store.save_messages(
            "batch_test",
            [SessionMessage(role="human", content="Second query")],
        )

        session = store.get_session("batch_test")
        assert len(session.messages) == 2
        assert session.query_count == 2

    def test_concurrent_stores_same_db(self, tmp_db):
        """Two store instances sharing the same DB should work (WAL mode)."""
        store1 = SessionStore(db_path=tmp_db)
        store2 = SessionStore(db_path=tmp_db)

        store1.create_session(
            session_id="shared1",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )

        # store2 should be able to read it
        session = store2.get_session("shared1")
        assert session is not None

    def test_special_characters_in_content(self, store):
        store.create_session(
            session_id="special_chars",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        messages = [
            SessionMessage(
                role="human",
                content="What's the bond angle in H₂O? Use O'Brien's method.",
            ),
            SessionMessage(
                role="ai",
                content='The angle is 104.5°. Here\'s the formula: "θ = 2·arcsin(d/2r)"',
            ),
        ]
        store.save_messages("special_chars", messages)

        session = store.get_session("special_chars")
        assert "O'Brien" in session.messages[0].content
        assert "104.5°" in session.messages[1].content

    def test_list_sessions_includes_message_count(self, store):
        store.create_session(
            session_id="msgcount",
            model_name="gpt-4o",
            workflow_type="single_agent",
        )
        store.save_messages(
            "msgcount",
            [
                SessionMessage(role="human", content="Q1"),
                SessionMessage(role="ai", content="A1"),
                SessionMessage(role="human", content="Q2"),
            ],
        )

        summaries = store.list_sessions()
        assert len(summaries) == 1
        assert summaries[0].message_count == 3
        assert summaries[0].query_count == 2
