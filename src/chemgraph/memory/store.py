"""
SQLite-based session storage for ChemGraph conversations.

Provides persistent storage for session metadata and message history,
enabling session listing, resumption, and context injection.
"""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from chemgraph.memory.schemas import Session, SessionMessage, SessionSummary

logger = logging.getLogger(__name__)

# Default database path: ~/.chemgraph/sessions.db
DEFAULT_DB_DIR = os.path.join(Path.home(), ".chemgraph")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "sessions.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    model_name   TEXT NOT NULL,
    workflow_type TEXT NOT NULL,
    log_dir      TEXT,
    query_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_name   TEXT,
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id);

CREATE INDEX IF NOT EXISTS idx_sessions_updated
    ON sessions(updated_at DESC);
"""


class SessionStore:
    """SQLite-backed persistent session store.

    Parameters
    ----------
    db_path : str, optional
        Path to SQLite database file. Defaults to ``~/.chemgraph/sessions.db``.
        The parent directory is created automatically if needed.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection with WAL mode and FK enforcement."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        model_name: str,
        workflow_type: str,
        title: str = "",
        log_dir: Optional[str] = None,
    ) -> Session:
        """Create a new session record.

        Parameters
        ----------
        session_id : str
            Unique session identifier (typically a UUID fragment).
        model_name : str
            LLM model name.
        workflow_type : str
            Workflow type (e.g., ``single_agent``).
        title : str, optional
            Human-readable title. Auto-generated later if empty.
        log_dir : str, optional
            Path to session log directory.

        Returns
        -------
        Session
            The newly created session.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (session_id, title, model_name, workflow_type, log_dir,
                     query_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (session_id, title, model_name, workflow_type, log_dir, now, now),
            )
        return Session(
            session_id=session_id,
            title=title,
            model_name=model_name,
            workflow_type=workflow_type,
            log_dir=log_dir,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )

    def save_messages(
        self,
        session_id: str,
        messages: list[SessionMessage],
        title: Optional[str] = None,
    ) -> None:
        """Append messages to a session and update metadata.

        Parameters
        ----------
        session_id : str
            Target session identifier.
        messages : list[SessionMessage]
            Messages to append.
        title : str, optional
            Update the session title (e.g., auto-generated from first query).
        """
        if not messages:
            return

        now = datetime.now().isoformat()
        human_count = sum(1 for m in messages if m.role == "human")

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO messages (session_id, role, content, tool_name, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        m.role,
                        m.content,
                        m.tool_name,
                        m.timestamp.isoformat(),
                    )
                    for m in messages
                ],
            )

            update_fields = ["updated_at = ?", "query_count = query_count + ?"]
            update_params: list = [now, human_count]

            if title:
                update_fields.append("title = ?")
                update_params.append(title)

            update_params.append(session_id)
            conn.execute(
                f"UPDATE sessions SET {', '.join(update_fields)} WHERE session_id = ?",
                update_params,
            )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Load a full session with all messages.

        Parameters
        ----------
        session_id : str
            Session identifier. Supports prefix matching if unique.

        Returns
        -------
        Session or None
            The session with messages populated, or None if not found.
        """
        resolved_id = self._resolve_session_id(session_id)
        if resolved_id is None:
            return None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (resolved_id,)
            ).fetchone()
            if not row:
                return None

            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                (resolved_id,),
            ).fetchall()

        messages = [
            SessionMessage(
                role=m["role"],
                content=m["content"],
                tool_name=m["tool_name"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
            )
            for m in msg_rows
        ]

        return Session(
            session_id=row["session_id"],
            title=row["title"],
            model_name=row["model_name"],
            workflow_type=row["workflow_type"],
            log_dir=row["log_dir"],
            query_count=row["query_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            messages=messages,
        )

    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SessionSummary]:
        """List sessions ordered by most recently updated.

        Parameters
        ----------
        limit : int
            Maximum number of sessions to return.
        offset : int
            Offset for pagination.

        Returns
        -------
        list[SessionSummary]
            Lightweight session summaries.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*,
                       (SELECT COUNT(*) FROM messages m
                        WHERE m.session_id = s.session_id) AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [
            SessionSummary(
                session_id=r["session_id"],
                title=r["title"],
                model_name=r["model_name"],
                workflow_type=r["workflow_type"],
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
                query_count=r["query_count"],
                message_count=r["message_count"],
            )
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Parameters
        ----------
        session_id : str
            Session identifier. Supports prefix matching.

        Returns
        -------
        bool
            True if a session was deleted, False if not found.
        """
        resolved_id = self._resolve_session_id(session_id)
        if resolved_id is None:
            return False

        with self._connect() as conn:
            # Messages are cascade-deleted via FK constraint
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (resolved_id,)
            )
            return cursor.rowcount > 0

    def session_count(self) -> int:
        """Return total number of stored sessions."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
            return row["cnt"]

    # ------------------------------------------------------------------
    # Context building for session resume
    # ------------------------------------------------------------------

    def build_context_messages(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        roles: Optional[list[str]] = None,
    ) -> list[dict]:
        """Build a list of message dicts suitable for injecting as LangGraph context.

        Extracts human, AI, and tool messages in chronological order.

        Parameters
        ----------
        session_id : str
            Session to extract context from.
        max_messages : int, optional
            Maximum number of messages to include (from the end).
        roles : list[str], optional
            Roles to include. Defaults to ``["human", "ai", "tool"]``.

        Returns
        -------
        list[dict]
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        session = self.get_session(session_id)
        if session is None:
            return []

        if roles is None:
            roles = ["human", "ai", "tool"]

        filtered = [m for m in session.messages if m.role in roles]

        if max_messages and len(filtered) > max_messages:
            filtered = filtered[-max_messages:]

        return [{"role": m.role, "content": m.content} for m in filtered]

    def build_context_summary(self, session_id: str) -> str:
        """Build a text summary of a previous session for context injection.

        This creates a concise summary that can be prepended to the system
        prompt or injected as a context message when resuming from a
        previous session.

        Parameters
        ----------
        session_id : str
            Session to summarize.

        Returns
        -------
        str
            A formatted summary string, or empty string if session not found.
        """
        session = self.get_session(session_id)
        if session is None:
            return ""

        human_msgs = [m for m in session.messages if m.role == "human"]

        lines = [
            "=== Previous Session Context ===",
            f"Session: {session.session_id}",
            f"Title: {session.title or 'Untitled'}",
            f"Model: {session.model_name}",
            f"Workflow: {session.workflow_type}",
            f"Date: {session.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Queries: {len(human_msgs)}",
            "",
            "Conversation:",
        ]

        for msg in session.messages:
            if msg.role == "human":
                lines.append(f"  User: {msg.content}")
            elif msg.role == "ai":
                # Truncate long AI responses for context
                content = msg.content
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"  Assistant: {content}")
            elif msg.role == "tool":
                tool_label = f" [{msg.tool_name}]" if msg.tool_name else ""
                content = msg.content
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"  Tool{tool_label}: {content}")

        lines.append("=== End Previous Session ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_session_id(self, session_id: str) -> Optional[str]:
        """Resolve a (possibly prefix) session ID to a full ID.

        Allows users to type just the first few characters of a UUID.
        Returns None if no match or ambiguous.
        """
        with self._connect() as conn:
            # Try exact match first
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return row["session_id"]

            # Try prefix match
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id LIKE ?",
                (session_id + "%",),
            ).fetchall()

            if len(rows) == 1:
                return rows[0]["session_id"]
            elif len(rows) > 1:
                logger.warning(
                    f"Ambiguous session ID prefix '{session_id}' matches "
                    f"{len(rows)} sessions. Please provide more characters."
                )
                return None
            return None

    @staticmethod
    def generate_title(query: str, max_length: int = 200) -> str:
        """Generate a session title from the first user query.

        Parameters
        ----------
        query : str
            The first user query.
        max_length : int
            Maximum title length.

        Returns
        -------
        str
            A cleaned-up title derived from the query.
        """
        title = query.strip()
        # Remove common prefixes
        for prefix in ["please ", "can you ", "could you ", "i want to ", "help me "]:
            if title.lower().startswith(prefix):
                title = title[len(prefix) :]
                break
        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]
        # Truncate
        if len(title) > max_length:
            title = title[: max_length - 3] + "..."
        return title
