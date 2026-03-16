#!/usr/bin/env python3
"""
Test script for ChemGraph session memory features.

Tests the full lifecycle:
  1. Run a query and verify session is created
  2. List sessions via SessionStore
  3. Show session details and messages
  4. Resume from the previous session with a follow-up query
  5. Verify resumed session has context injected
  6. Clean up test sessions

Usage:
    python scripts/test_session_memory.py

Requires:
    - Argo API access (uses gpt4o via Argo endpoint)
    - ARGO_USER env var or defaults to 'chemgraph'
"""

import asyncio
import sys
import os

# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt4o"
BASE_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
WORKFLOW_TYPE = "single_agent"
# ---------------------------------------------------------------------------


async def main():
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.memory.store import SessionStore

    # Use a temp database so we don't pollute the real session store
    import tempfile

    tmp_db = os.path.join(tempfile.mkdtemp(), "test_sessions.db")
    print(f"Using temp database: {tmp_db}\n")

    # ------------------------------------------------------------------
    # Step 1: First run — create a session
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: First run — Calculate thermochemistry of CO2")
    print("=" * 60)

    cg = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type=WORKFLOW_TYPE,
        structured_output=False,
        return_option="state",
        base_url=BASE_URL,
        memory_db_path=tmp_db,
    )

    print(f"Agent UUID:      {cg.uuid}")
    print(f"Agent session_id:{cg.session_id}")
    print(f"Log dir:         {cg.log_dir}")
    print()

    # Visualize the workflow graph
    print("Workflow graph:")
    print(cg.visualize())
    print()

    query1 = "Calculate the thermochemistry of CO2 at 298K using Mace_mp, medium model"
    print(f"Query: {query1}\n")

    result1 = await cg.run(query1, {"thread_id": 1})
    session1_id = cg.session_id

    print(f"\nSession 1 ID: {session1_id}")
    print(f"Session created: {cg._session_created}")

    # ------------------------------------------------------------------
    # Step 2: List sessions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: List sessions")
    print("=" * 60)

    store = SessionStore(db_path=tmp_db)
    sessions = store.list_sessions()

    print(f"Total sessions: {store.session_count()}")
    for s in sessions:
        print(
            f"  [{s.session_id}] {s.title} "
            f"| model={s.model_name} "
            f"| queries={s.query_count} "
            f"| messages={s.message_count} "
            f"| {s.updated_at.strftime('%Y-%m-%d %H:%M')}"
        )

    # ------------------------------------------------------------------
    # Step 3: Show session details
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3: Show session {session1_id}")
    print("=" * 60)

    session = store.get_session(session1_id)
    if session:
        print(f"Title:    {session.title}")
        print(f"Model:    {session.model_name}")
        print(f"Workflow: {session.workflow_type}")
        print(f"Log dir:  {session.log_dir}")
        print(f"Messages: {len(session.messages)}")
        print()
        for msg in session.messages:
            role_label = {"human": "User", "ai": "Assistant", "tool": "Tool"}.get(
                msg.role, msg.role
            )
            tool_suffix = f" [{msg.tool_name}]" if msg.tool_name else ""
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"  {role_label}{tool_suffix}: {content}")
    else:
        print(f"ERROR: Session {session1_id} not found!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Build context summary (what --resume injects)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Context summary (preview of what --resume injects)")
    print("=" * 60)

    summary = store.build_context_summary(session1_id)
    print(summary)

    # ------------------------------------------------------------------
    # Step 5: Resume from session 1 with a follow-up query
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Resume — follow-up query using previous session context")
    print("=" * 60)

    # Clear CHEMGRAPH_LOG_DIR so second agent creates its own log dir
    if "CHEMGRAPH_LOG_DIR" in os.environ:
        del os.environ["CHEMGRAPH_LOG_DIR"]

    cg2 = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type=WORKFLOW_TYPE,
        structured_output=False,
        return_option="state",
        base_url=BASE_URL,
        memory_db_path=tmp_db,
    )

    query2 = "Now calculate the same thermochemistry but at 500K instead"
    print(f"Query: {query2}")
    print(f"Resuming from session: {session1_id}\n")

    result2 = await cg2.run(query2, {"thread_id": 1}, resume_from=session1_id)
    session2_id = cg2.session_id

    print(f"\nSession 2 ID: {session2_id}")

    # ------------------------------------------------------------------
    # Step 6: Verify both sessions exist
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Final session listing")
    print("=" * 60)

    sessions = store.list_sessions()
    print(f"Total sessions: {store.session_count()}")
    for s in sessions:
        print(
            f"  [{s.session_id}] {s.title} "
            f"| queries={s.query_count} "
            f"| messages={s.message_count}"
        )

    # ------------------------------------------------------------------
    # Step 7: Test prefix matching
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Prefix matching")
    print("=" * 60)

    prefix = session1_id[:4]
    resolved = store.get_session(prefix)
    if resolved:
        print(f"Prefix '{prefix}' resolved to: {resolved.session_id}")
    else:
        print(f"Prefix '{prefix}' did not resolve (may be ambiguous)")

    # ------------------------------------------------------------------
    # Step 8: Test load_previous_context from agent API
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: load_previous_context() via agent API")
    print("=" * 60)

    context = cg2.load_previous_context(session1_id)
    if context:
        # Show first 500 chars
        preview = context[:500] + "..." if len(context) > 500 else context
        print(f"Context loaded ({len(context)} chars):")
        print(preview)
    else:
        print("WARNING: No context returned")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Session 1: {session1_id} (initial query)")
    print(f"Session 2: {session2_id} (resumed from session 1)")
    print(f"Database:  {tmp_db}")
    print(f"Log dir 1: {cg.log_dir}")
    print(f"Log dir 2: {cg2.log_dir}")
    print()
    print("To explore further with the CLI:")
    print(f"  chemgraph --list-sessions")
    print(f"  chemgraph --show-session {session1_id}")
    print(f"  chemgraph -q 'Your query' --resume {session1_id}")
    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
