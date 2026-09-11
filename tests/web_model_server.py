"""Test-only OpenAI-compatible endpoint. Never imported by the runtime package."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re


def reply(body):
    messages = body["messages"]
    text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message["role"] != "system"
    )
    system = str(messages[0].get("content", ""))
    tools = {tool["function"]["name"] for tool in body.get("tools", [])}

    def call(name, arguments):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"test_{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    if "Follow-up:" in text:
        content = (
            "Previous context retained: copper and EMT."
            if "copper" in text.lower() and "EMT" in text
            else "Missing previous context."
        )
        if "**Planner**" in system:
            content = json.dumps(
                {"next_step": "FINISH", "thought_process": content, "tasks": []}
            )
        return {"role": "assistant", "content": content}

    needs_confirmation = (
        "confirm" in text.lower() and "yes, continue" not in text.lower()
    )
    if "**Planner**" in system:
        if "UPDATED: Results from Executor Tasks" in text:
            result = text.split("UPDATED: Results from Executor Tasks ###", 1)[1]
            response = {"next_step": "FINISH", "thought_process": result, "tasks": []}
        elif needs_confirmation:
            response = {
                "next_step": "ask_human",
                "thought_process": "Request confirmation",
                "clarification": "Optimize the attached copper with EMT?",
                "tasks": [],
            }
        else:
            response = {
                "next_step": "executor_subgraph",
                "thought_process": "Run EMT",
                "tasks": [{"task_index": 1, "prompt": text}],
            }
        return {"role": "assistant", "content": json.dumps(response)}

    if needs_confirmation and "ask_human" in tools:
        return call("ask_human", {"question": "Optimize the attached copper with EMT?"})
    tool_results = [
        message
        for message in messages
        if message["role"] == "tool" and message.get("tool_call_id") == "test_run_ase"
    ]
    # The existing single-agent graph supplies its transcript as text, while
    # multi-agent executors send native role/tool messages.
    energies = re.findall(r'"potential_energy":\s*([0-9.e+\-]+)', text)
    if not tool_results and "ToolMessage" in text and energies:
        return {
            "role": "assistant",
            "content": f"Calculation complete. EMT potential energy: **{float(energies[-1]):.4f} eV**.",
        }
    if tool_results:
        result = json.loads(tool_results[-1]["content"])
        energy = result.get("potential_energy")
        if energy is None:
            raise ValueError("EMT result did not contain potential energy")
        return {
            "role": "assistant",
            "content": f"Calculation complete. EMT potential energy: **{energy:.4f} eV**.",
        }
    words = text.replace("\\", " ").replace('"', " ").replace("'", " ").split()
    attachment = next(
        (
            word[word.index("/") :]
            for word in words
            if "/uploads/" in word and word.endswith(".xyz")
        ),
        None,
    )
    if attachment is None:
        raise ValueError("Test requires an attached XYZ structure")
    return call(
        "run_ase",
        {
            "ase_input": {
                "input_structure_file": attachment,
                "output_results_file": "copper-result.json",
                "driver": "opt",
                "steps": 10,
                "fmax": 0.05,
                "calculator": {"calculator_type": "emt"},
            }
        },
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model", "").startswith("fail-"):
            status = int(body["model"].split("-", 1)[1])
            payload = {
                "error": {
                    "message": "PRIVATE_PROVIDER_RESPONSE_MARKER",
                    "type": "provider_error",
                    "code": "test_failure",
                }
            }
        else:
            try:
                message = reply(body)
                status = 200
                payload = {
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls"
                            if message.get("tool_calls")
                            else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            except Exception as exc:
                status, payload = 400, {"error": {"message": str(exc)}}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
