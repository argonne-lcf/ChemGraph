"""Single-query container entry point. No controller configuration is accepted."""

import asyncio
import json
import os
from pathlib import Path
import sys

from chemgraph.eval.protocol import MAX_REQUEST_BYTES, PROTOCOL_VERSION, RESULT_FILE, WorkerRequest

PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")


def main():
    data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("Worker request exceeds size limit")
    request = WorkerRequest.model_validate_json(data)
    # Redirect library/native prints as well as Python logging. The result travels
    # in a regular file collected only after the container has stopped.
    log = os.open("/tmp/worker.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    from chemgraph.eval.deepagent import execute_query

    raw = asyncio.run(execute_query(
        request.execution, request.model, request.query, request.query_id,
        workspace_path=Path("/workspace"), base_url=request.base_url, argo_user=request.argo_user,
        shell_environment={key: os.environ[key] for key in (
            *PROXY_ENV_KEYS, "HOME", "XDG_CACHE_HOME", "PYTHONNOUSERSITE", "OMP_NUM_THREADS",
            "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
        ) if key in os.environ},
    ))
    payload = json.dumps({"version": PROTOCOL_VERSION, "raw": raw}, allow_nan=False)
    # Refuse links planted by executed tools.
    fd = os.open(f"/workspace/{RESULT_FILE}", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(payload)


if __name__ == "__main__":
    main()
