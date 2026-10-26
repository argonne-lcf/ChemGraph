# Source from bash: IRI API v2 helpers for the iri-hpc skill.
# The token reaches curl only through a pipe from the builtin printf, and -q keeps
# a user's curlrc from enabling tracing or redirects.
set -o pipefail
BASE=${BASE:-https://api.alcf.anl.gov/api/v2}

# Load IDs and paths from the workspace's target.toml, so commands never retype
# them; a variable the caller already set wins.
if [ -f target.toml ]; then
  if _iri_target=$(python - target.toml <<'PY'
import shlex, sys, tomllib
names = {
    "compute_resource_id": "COMPUTE", "storage_resource_id": "STORAGE",
    "username": "OWNER", "account": "ACCOUNT", "remote_root": "REMOTE_ROOT",
    "local_collection_id": "LOCAL", "remote_collection_id": "REMOTE_COLLECTION",
    "local_globus_root": "LOCAL_ROOT", "remote_globus_root": "REMOTE_GLOBUS_ROOT",
}
with open(sys.argv[1], "rb") as handle:
    target = tomllib.load(handle)
for key, name in names.items():
    if key in target:
        print(f'[ -n "${{{name}-}}" ] || {name}={shlex.quote(str(target[key]))}')
PY
  ); then
    eval "$_iri_target"
    unset _iri_target
  else
    unset _iri_target
    echo "Could not read target.toml; fix or recreate it before continuing." >&2
    return 1
  fi
fi

iri() {
  set +x
  local token
  token=$(alcf-tokens get-token iri) && [ -n "$token" ] ||
    { echo "IRI token unavailable; ask the user to run: alcf-tokens login iri" >&2; return 1; }
  local out rc
  out=$(printf 'Authorization: Bearer %s\n' "$token" | curl -q -sS --fail-with-body -H @- "$@")
  rc=$?
  # Callers usually pipe stdout into jq, so repeat an error body on stderr.
  if [ "$rc" -ne 0 ]; then
    [ -n "$out" ] && printf '%s\n' "$out" >&2
    return "$rc"
  fi
  [ -n "$out" ] && printf '%s\n' "$out"
  return 0
}

# POST a JSON body to a filesystem operation and wait for its task.
iri_fs() {  # iri_fs OPERATION JSON
  local task
  : "${STORAGE:?run commands from the workspace that holds target.toml (iri-hpc setup writes it)}"
  task=$(iri -X POST -H 'Content-Type: application/json' -d "$2" \
    "$BASE/filesystem/$1/$STORAGE" | jq -er '.task_id') || return 1
  iri_task "$task"
}

# Print a finished task; fail on failed/canceled tasks or after about 5 minutes.
iri_task() {  # iri_task TASK_ID
  local task state=unknown
  for _ in $(seq 60); do
    task=$(iri "$BASE/task/$1") || return 1
    state=$(jq -er '.status' <<<"$task") || return 1
    case $state in
      completed) printf '%s\n' "$task"; return 0 ;;
      failed|canceled) printf '%s\n' "$task" >&2; return 1 ;;
    esac
    sleep 5
  done
  echo "Task $1 is still $state" >&2
  return 1
}
