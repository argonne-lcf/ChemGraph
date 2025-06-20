import chainlit as cl
import os
import json

from cg_ui.tasks import run_workflow_async
from cg_ui.components import mol_viewer, log_tail


@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content="Welcome to ChemGraph! What would you like to calculate?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    prompt = message.content
    msg = cl.Message(content="", author="ChemGraph")
    await msg.send()

    session_log = []
    full_response = ""
    final_result = None

    async for chunk in run_workflow_async(prompt):
        if "token" in chunk and chunk["token"]:
            full_response += chunk["token"]
            await msg.stream_token(chunk["token"])

        if "log" in chunk:
            session_log.append(chunk["log"])

        if chunk.get("status") == "done":
            final_result = chunk.get("result", {})
            if "log" in chunk:
                session_log.append(f"Final Log:\n{chunk['log']}")

    if not full_response.strip():
        full_response = "The workflow completed its task."

    msg.content = full_response
    await msg.update()

    if final_result:
        # Build the final message content
        final_content_parts = ["**Final Result**"]

        # Display final answer from structured output
        if "answer" in final_result:
            answer_data = final_result["answer"]
            if isinstance(answer_data, dict):
                # We handle the geometry display below, so don't show the raw dict
                if "numbers" not in answer_data and "positions" not in answer_data:
                    final_content_parts.append(
                        f"```json\n{json.dumps(answer_data, indent=2)}\n```"
                    )
            else:
                final_content_parts.append(str(answer_data))
        else:
            final_content_parts.append("Workflow finished.")

        # Display molecule if present
        if "mol_xyz" in final_result:
            final_content_parts.append(mol_viewer(final_result["mol_xyz"]))

        final_answer_str = "\n\n".join(final_content_parts)

        # Add a collapsible log view
        if session_log:
            log_content = "\n".join(session_log)
            final_answer_str += f"\n\n{log_tail(log_content)}"

        await cl.Message(content=final_answer_str, author="ChemGraph").send()


def main():
    """
    Entry point for the chemgraph-ui script.

    This function provides instructions on how to run the Chainlit application,
    as it cannot be started directly as a standard Python script.
    """
    print("To run the ChemGraph UI, use the command:")
    print("\n  python run_ui.py\n")
