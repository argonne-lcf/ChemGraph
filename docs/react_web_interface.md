# React interface (internal pilot)

ChemGraph's React interface runs as a separate frontend and Python API. It
provides chat, administrator-approved model selection, single-/multi-agent
workflows, structure/data attachments, live tool progress, follow-up questions,
saved conversations, 3D structures, trajectory playback, and artifact downloads.
The existing Streamlit interface and `chemgraph ui` command remain available.

## Run a local demo with Docker Compose

From a source checkout:

```bash
CHEMGRAPH_WEB_DEV_USER=local-demo CHEMGRAPH_WEB_DEMO=true \
  docker compose -f compose.web.yml up --build -d
```

Open <http://localhost:8080>. The demo runs a real, small copper-dimer EMT
optimization without model credentials. Include `confirm` in a message to
exercise a follow-up question. Demo execution always uses this fixed example;
it does not interpret the requested chemistry or simulate uploaded structures.

The frontend is bound to loopback. The API has no published host port. Data is
stored in the `web-data` named volume; stopping/recreating the containers keeps
it. `docker compose -f compose.web.yml down` stops the deployment without
deleting its volumes. Do not use `down --volumes` to preserve conversations.

The API image extends the maintained ChemGraph runtime and installs the current
checkout. Set `CHEMGRAPH_BASE_IMAGE` to an approved immutable image digest for
your deployment; the base must provide this checkout's core Python dependencies.
The frontend image contains static assets and Nginx, with no Python runtime.

## Develop without Docker

The API runtime supports Linux and macOS. Production packaging targets Linux.
Use Python 3.11+ and Node.js 22.12+:

```bash
python -m pip install -e '.[web]'
CHEMGRAPH_WEB_DEV_USER=local-demo CHEMGRAPH_WEB_DEMO=true \
  python -m chemgraph.api
```

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:8080>; Vite proxies `/api/` to port 8000. Keep the browser
origin consistent with `CHEMGRAPH_WEB_PUBLIC_ORIGIN` (default
`http://localhost:8080`), including hostname and port. For example, opening
`127.0.0.1` instead requires setting that origin on the API.

## Connect approved models

Administrators set `CHEMGRAPH_WEB_PROVIDERS` to a JSON object whose keys are the
model labels shown to users. A definition accepts `model`, optional `base_url`,
optional `api_key_env`, and optional `argo_user`. Keys and endpoint definitions
stay on the server. Example non-secret configuration:

```json
{
  "OpenAI · GPT-4o mini": {
    "model": "gpt-4o-mini",
    "api_key_env": "OPENAI_API_KEY"
  },
  "Lab endpoint": {
    "model": "lab-model",
    "base_url": "https://inference.example.org/v1",
    "api_key_env": "LAB_MODEL_API_KEY"
  }
}
```

Supply the referenced environment variables through your secret manager. Add
custom credential variables to the Compose/Kubernetes API environment as needed.
Disable `CHEMGRAPH_WEB_DEMO` to use approved providers. A missing explicitly
configured credential produces a service-unavailable error before queueing work.
Each conversation keeps the model/workflow chosen at creation; start a new
conversation to change them. Removing a model disables new runs in its old
conversations while retaining their history.

## Deploy behind institutional SSO

For a shared service, **unset `CHEMGRAPH_WEB_DEV_USER` and disable demo mode**.
The development user bypasses gateway authentication and is for private local
testing only. Route both `/` and `/api/` through the same authenticated origin.

The gateway must remove any client-supplied `X-Auth-Request-User` header and
insert the verified, stable user identifier after login. If your gateway uses
a different header, configure `CHEMGRAPH_WEB_IDENTITY_HEADER`. Use a stable
subject identifier; changing the identity makes its existing conversations
unavailable to that user. The API hashes it for ownership checks.

Set `CHEMGRAPH_WEB_PUBLIC_ORIGIN` to the exact HTTPS origin. API mutations require
that `Origin` header; browser `fetch` supplies it. Scripts must include it too.
Restrict network access to the frontend to the gateway and API access to the
frontend. Header-based identity is valid only within this trusted boundary.
The supplied Nginx proxy disables buffering for SSE and does not execute HTML
reports in the application origin; artifact downloads use attachment responses.

For Kubernetes, customize the templates in `k8s/web/` before applying:

1. Build/tag/push the two images into your registry and set their image digests.
2. Set the hostname, public origin, TLS secret, provider definitions, and
   credential secret references.
3. Replace the ingress external-auth URLs and configure the authenticated
   identity header. Match the gateway namespace in the frontend NetworkPolicy.
4. Choose a persistent volume/storage class with reliable POSIX file locking
   for SQLite (a block-backed volume rather than an NFS share). Verify that the
   cluster network plugin enforces the supplied NetworkPolicies.
5. Inspect `kubectl kustomize k8s/web`, then apply it in your pilot namespace:

```bash
kubectl apply -k k8s/web -n YOUR_NAMESPACE
```

The API uses **one replica and one Uvicorn process**, with a `Recreate` rollout
strategy. An exclusive lock prevents two API processes from sharing its data
directory. Its `/healthz` probe checks storage and scheduler health. The static
frontend is independently deployable. Existing Streamlit/MCP deployments are
not changed by these manifests.

## Execution, storage, and limits

The API stores HTTP resources and replayable events in `web.db`. A separate
`SessionStore` database retains each conversation's agent context. Files live
under session/run directories. These stores are separate from Streamlit's
existing sessions; automatic import is not part of the pilot.

Workers run in separate processes to isolate environment variables and avoid
blocking HTTP requests. The web tool set excludes Python/shell execution,
arbitrary model-file loading, and arbitrary filesystem paths. Tool reads stay
inside their conversation; new outputs stay inside the current run. The pilot
allows EMT by default. `CHEMGRAPH_WEB_CALCULATORS` can enable `mace_mp`,
`mace_off`, or `mace_polar` (JSON array); provision those foundation models and
their dependencies on the server. Only their default model selections are
exposed. This remains an internal pilot, not a sandbox for untrusted code.

| Setting | Default | Meaning |
| --- | --- | --- |
| `CHEMGRAPH_WEB_DATA_DIR` | `web_data` (`/data` in Docker) | Persistent data root |
| `CHEMGRAPH_WEB_MAX_WORKERS` | `2` | Maximum simultaneous worker processes |
| `CHEMGRAPH_WEB_RUN_TIMEOUT` | `3600` | Seconds per worker, including time waiting for human input |
| `CHEMGRAPH_WEB_UPLOAD_LIMIT` | `26214400` | Maximum bytes per attachment; align proxy limits if changed |

Only one unfinished run is allowed in each conversation. Other conversations
queue while workers are occupied. Each message accepts up to ten XYZ, PDB, CIF,
TRAJ, JSON, CSV, or TXT attachments. Structure previews are limited to 500 frames,
20,000 atoms/frame, and 25 MiB of XYZ; larger artifacts can still be downloaded.

Refreshing or closing a browser does not stop work. Reopening its conversation
reconnects to status and events without resubmitting the calculation. A backend
restart retains finished history and files but marks unfinished/paused runs as
interrupted. Retrying is explicit and starts a new run. Completed conversations
continue with the existing stored-context summary mechanism, not restoration
of a prior live graph checkpoint. Save/back up the data volume between upgrades.

## HTTP interface and tests

The authenticated API schema is at `/api/v1/openapi.json`. Endpoints cover
capabilities, sessions, binary uploads, run submission/status, human responses,
SSE events, and artifacts. Submissions include a UUID `request_id` for retry
deduplication. Human responses reference the pending `question_id`. Artifact IDs
are opaque; the API never accepts a server path as a download target. SSE
supports `Last-Event-ID` and `after` replay cursors. This release streams tool and
workflow progress rather than individual model tokens.

```bash
ruff check .
pytest tests/ -k 'not tblite'
cd frontend
npm test
npm run build
```

With the local demo running, `npm run test:e2e` tests attachments, follow-up
questions, refresh/reconnect, artifact downloads, WebGL structure rendering,
trajectory controls, and mobile panels. Install the Playwright Chromium runtime
for CI, or set `WEB_TEST_BROWSER=chrome` to use an installed Chrome browser.
Set `WEB_TEST_URL` when testing a different frontend origin.

Personal provider login, advanced IR exploration, arbitrary Python workflows,
durable recovery of unfinished graphs, and horizontal API scaling remain in
future work. The pilot does not change the existing Streamlit feature set.
