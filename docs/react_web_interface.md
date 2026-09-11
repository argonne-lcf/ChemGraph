# React interface for institutional users

ChemGraph's React interface runs as a separate frontend and Python API. It
provides chat, administrator-approved model selection, single-/multi-agent
workflows, structure/data attachments, live tool progress, follow-up questions,
saved conversations, 3D structures, trajectory playback, and artifact downloads.
The existing Streamlit interface and `chemgraph ui` command remain available.

The first production deployment is for internal users behind institutional SSO,
using shared administrator-managed Argo or ALCF credentials. SSO controls data
ownership; all users' provider calls use the configured shared account. Per-user
provider identities and token management are future work. Hosted endpoints do
not require another container.

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

Copy `web-providers.example.toml` and select approved models:

```toml
[providers.Argo]
model = "argo:gpt-4o"
# Uses the shared ARGO_USER environment value.

[providers.ALCF]
model = "alcf:meta-llama/Llama-3.3-70B-Instruct"
api_key_env = "ALCF_ACCESS_TOKEN"
```

For a local Python API, set `CHEMGRAPH_WEB_PROVIDERS_FILE` to the file path. For
Compose, set `CHEMGRAPH_WEB_PROVIDERS_HOST_FILE` to the host file path; Compose
mounts it read-only at `/etc/chemgraph/providers.toml` and sets the API's selector.
The example file is mounted by default. Keep demo disabled for real LLM runs.

The existing `CHEMGRAPH_WEB_PROVIDERS` JSON object remains supported. When set,
it completely replaces the file's list, without reading the file. An empty
string is invalid; `{}` deliberately disables all models. Compose leaves this
variable absent unless supplied. Unknown fields and literal API-key fields are
rejected. URLs cannot contain embedded credentials, query strings, or fragments.

Argo resolves `argo_user` first, then `ARGO_USER`. A real shared identity must be
configured; the web service does not use the generic `chemgraph` identity.
ALCF uses `api_key_env` if supplied, otherwise `ALCF_ACCESS_TOKEN`. A missing
explicit override never falls back to another variable. Other hosted providers
retain their provider-specific credential variables. Custom OpenAI-compatible
routes use an explicit variable, `VLLM_API_KEY`, or a no-auth placeholder; they
never inherit `OPENAI_API_KEY`. Subscription authentication is unavailable here.

Supply credentials through deployment secrets or the API environment. Compose
forwards `ARGO_USER`, `ALCF_ACCESS_TOKEN`, and the documented hosted-provider
variables; add any custom `api_key_env` variable to the API service's environment
in a local Compose override. Host environment variables are not automatically
available inside containers. Never put credentials in frontend/Vite variables.

Validate without starting workers, creating application storage, prompting, or
contacting a provider:

```bash
python -m chemgraph.api --check-config
# Or validate the exact container environment and mounted file:
docker compose -f compose.web.yml run --rm --no-deps web-api python -m chemgraph.api --check-config
```

Exit codes: `0` means every configured model has its required configuration;
`1` means missing credentials or no models; `2` means invalid configuration.
Malformed configuration stops startup. Missing credentials leave the API and
history available, with the affected model disabled. "Configured" does not
mean that connectivity or credential validity has been verified.

File edits and environment-based credential rotation require an API restart.
Conversations retain a non-secret fingerprint of their model and endpoint.
Changing either, removing a model, or disabling demo makes affected conversations
read-only. Credential rotation does not change the fingerprint. Conversations
created by the earlier pilot without a fingerprint also become read-only.
Their history remains readable to their original owner. Development-user
history is not reassigned to an SSO identity during cutover.
Upgraded conversations receive a full retention window starting at migration,
because the earlier schema did not record activity. Previously exposed internal
graph-state download IDs are revoked; chemistry outputs remain available.

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
2. Set the hostname, public origin, and TLS secret. Edit `k8s/web/providers.toml`;
   Kustomize generates a ConfigMap and mounts it read-only. Supply the
   `argo-user` and/or `alcf-access-token` keys in `chemgraph-secrets` for the
   enabled providers. Missing keys leave those providers unavailable.
3. Replace the ingress external-auth URLs and configure the authenticated
   identity header. Match the gateway namespace in the frontend NetworkPolicy.
   Use an institution-supported controller: the included ingress-nginx
   annotations illustrate the authentication contract, but upstream
   [ingress-nginx was retired in March 2026](https://kubernetes.io/blog/2025/11/11/ingress-nginx-retirement/)
   and is not a new deployment default.
   Translate those settings to the gateway supported by your institution.
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
| `CHEMGRAPH_WEB_PROVIDER_TIMEOUT` | `120` | Timeout in seconds per provider request |
| `CHEMGRAPH_WEB_MAX_QUEUED` | `20` | Maximum queued runs across users |
| `CHEMGRAPH_WEB_MAX_USER_RUNS` | `2` | Maximum unfinished runs per user |
| `CHEMGRAPH_WEB_CLARIFICATION_TIMEOUT` | `900` | Maximum seconds awaiting a clarification response |
| `CHEMGRAPH_WEB_USER_STORAGE_LIMIT` | `1073741824` | Per-user workspace, memory, diagnostics, and transcript byte budget |
| `CHEMGRAPH_WEB_RETENTION_DAYS` | `30` | Retention after the last write activity of inactive conversations |
| `CHEMGRAPH_WEB_CONTEXT_LIMIT` | `16000` | Maximum characters of prior context sent on a follow-up |

Only one unfinished run is allowed in each conversation. Other conversations
queue while workers are occupied. Each message accepts up to ten XYZ, PDB, CIF,
TRAJ, JSON, CSV, or TXT attachments. Structure previews are limited to 500 frames,
20,000 atoms/frame, and 25 MiB of XYZ; larger artifacts can still be downloaded.
The approved reader handles those structure and text formats; text content sent
to the model is capped at 50,000 characters. Only one worker executes per user,
including clarification waits. Other users can use the remaining worker slots.

The UI can cancel queued, running, or waiting runs. A running cancellation stays
in `cancelling` until the worker is stopped; then it becomes `cancelled` and
releases capacity. Whole-run retries are never automatic. Existing SDK retries
and graph parse/task retries remain; a request timeout can therefore consume
more than one attempt, bounded by the worker timeout. The default graph allows
one parse retry and multi-agent execution allows up to two task attempts.

Uploads reserve space before writing. Chemistry tools check storage around their
work and the supervisor interrupts workers that exceed the budget. This is an
application quota, not a filesystem hard limit; a tool can briefly exceed it
between checks. Provision filesystem/container limits and monitor total volume
usage, including shared model caches and SQLite/WAL overhead. Hourly retention
cleanup removes inactive conversations and their artifacts, diagnostics, and
agent memory; active runs and in-progress uploads are excluded.

Refreshing or closing a browser does not stop work. Reopening its conversation
reconnects to status and events without resubmitting the calculation. A backend
restart retains finished history and files but marks unfinished/paused runs as
interrupted. Retrying is explicit and starts a new run. Completed conversations
continue with the existing stored-context summary mechanism, not restoration
of a prior live graph checkpoint. Context uses a bounded tail and the stored
user messages do not include recursively injected summaries. Explicit retries
retain the original attachments. Internal graph-state JSON is stored separately
and is never registered as a chemistry download.

## Release and operations

Before opening access, verify the actual gateway: unauthenticated requests and
forged identity headers cannot reach user data, two test users cannot access
each other's sessions/downloads, direct API access is blocked, TLS is valid,
and SSE reconnect works after browser refresh and SSO renewal. These are site
acceptance checks; local tests cannot verify an institution's SSO configuration.

Run the optional provider smoke test from the deployed API environment for
every enabled provider before release and after endpoint or credential changes.
Mount a copy of the test directory into a temporary container if it is not in
the image. It performs a real, small EMT calculation through the selected LLM:

```bash
pytest tests/test_web_integration.py::test_enabled_live_providers_smoke --run-llm -q
```

Normal CI does not enable this flag. A present ALCF token can still be expired;
rotate it through the institution's approved process and restart the API.
There is no browser credential entry and no automatic token refresh in v1.
Shared-account usage, attribution, and quotas should be approved by the site.

Monitor `/healthz`, provider error codes and run IDs, queue age, interrupted runs,
and storage usage. Keep provider exception logs access-restricted. For backup,
stop the API and copy/snapshot the entire data volume, including `web.db`, its
WAL files if present, session files, diagnostics, and memory databases. Restore
the complete snapshot into an empty volume with the API stopped. Upgrade after
a backup; configuration/schema changes are startup-only. Rollback restores both
the previous image and its matching data snapshot. Restarted runs are interrupted
and require an explicit retry; one API replica and `Recreate` rollouts remain.

## HTTP interface and tests

The authenticated API schema is at `/api/v1/openapi.json`. Endpoints cover
capabilities, sessions, binary uploads, run submission/status, human responses,
SSE events, cancellation, and artifacts. Submissions include a UUID `request_id` for retry
deduplication. Human responses reference the pending `question_id`. Artifact IDs
are opaque; the API never accepts a server path as a download target. SSE
supports `Last-Event-ID` and `after` replay cursors. This release streams tool and
workflow progress rather than individual model tokens.
`capabilities.models` remains a label list; `model_status` adds configuration
readiness and safe reasons. Session details include readiness for that specific
conversation. Definitive API rejections include `submission_rejected: true`;
transport failures or unclassified proxy errors retain the same request ID for
an uncertain submission. Cancel with `POST /api/v1/runs/{id}/cancel`.

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

The deterministic non-demo endpoint lives only in `tests/web_model_server.py`.
`compose.web.test.yml` builds an isolated API, frontend, and test model endpoint;
it mounts test configuration rather than deployment secrets:

```bash
docker compose -p chemgraph-web-test -f compose.web.test.yml up --build -d --wait
cd frontend
WEB_TEST_REAL=true WEB_TEST_URL=http://localhost:8082 npm run test:e2e
```

Stop this test deployment from the repository root with
`docker compose -p chemgraph-web-test -f compose.web.test.yml down --volumes`.
Its volumes contain only disposable test data. Browser CI runs demo mode and
real graphs in Vite, then repeats the real workflow through the built Nginx proxy.

Personal provider login, advanced IR exploration, arbitrary Python workflows,
durable recovery of unfinished graphs, and horizontal API scaling remain in
future work. The pilot does not change the existing Streamlit feature set.
