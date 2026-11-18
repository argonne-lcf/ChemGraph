"""Bounded conversation context without recursively persisting injected history."""

from chemgraph.memory.store import SessionStore


class WebSessionStore(SessionStore):
    def __init__(self, path, original_query, context_limit):
        super().__init__(path)
        self.original_query = original_query
        self.context_limit = context_limit

    def build_context_summary(self, session_id):
        # Query a bounded tail without loading an entire long-lived transcript.
        with self._connect() as db:
            rows = db.execute(
                "SELECT role,substr(content,1,2000) AS content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 40",
                (session_id,),
            ).fetchall()
        context = "\n".join(
            f"{row['role']}: {row['content']}" for row in reversed(rows)
        )
        return context[-self.context_limit :]

    def save_messages(self, session_id, messages, **kwargs):
        # Each web turn starts a new graph; its first human message contains
        # ephemeral history and attachment instructions. Persist the user input.
        messages = list(messages)
        for index, message in enumerate(messages):
            if message.role == "human":
                messages[index] = message.model_copy(
                    update={"content": self.original_query}
                )
                break
        return super().save_messages(session_id, messages, **kwargs)
