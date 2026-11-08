"""Offline TLS Argo/PubChem fixture. Installed only in explicit test images."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import ssl


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply({"PropertyTable": {"Properties": [{"CID": 962, "ConnectivitySMILES": "O", "SMILES": "O"}]}})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/messages"):
            result = {"id": "msg_test", "type": "message", "role": "assistant",
                      "model": request["model"], "content": [{"type": "text", "text": '{"smiles":["O"]}'}],
                      "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}
        else:
            result = {"id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                      "model": request["model"], "choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": '{"smiles":["O"]}'}}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        if request.get("stream") and not self.path.endswith("/messages"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0,
                     "model": request["model"], "choices": [{"index": 0, "finish_reason": None,
                     "delta": {"role": "assistant", "content": '{"smiles":["O"]}'}}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            chunk["choices"][0].update(delta={}, finish_reason="stop")
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
        else:
            self.reply(result)

    def reply(self, result):
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


server = ThreadingHTTPServer(("0.0.0.0", 443), Handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain("/opt/test.crt", "/opt/test.key")
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
