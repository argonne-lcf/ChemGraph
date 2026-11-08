"""Optional rootless Podman evaluation controller; never falls back to local execution."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tarfile
import tempfile
import time
import uuid

from chemgraph.agent.usage import apply_history_coverage, summarize_usage
from chemgraph.eval.protocol import (
    ExecutionSettings, MAX_RESULT_BYTES, RESULT_FILE, WorkerRequest, validate_response,
)

LABEL = "org.chemgraph.evaluation.run"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
POLICY = {
    "version": 1, "rootless": True, "network": "internal-bridge-and-connect-proxy",
    "connect_hosts": ["apps.inside.anl.gov", "pubchem.ncbi.nlm.nih.gov"], "connect_port": 443,
    "read_only": True, "user": "1000:1000", "cap_drop": "ALL",
    "no_new_privileges": True, "seccomp": "default", "pids_limit": 256,
    "mounts": ["private-volume:/workspace", "private-tmpfs:/tmp", "private-tmpfs:/run"],
    "memory": False, "discover_skills": False,
    "max_archive_bytes": MAX_ARCHIVE_BYTES, "max_result_bytes": MAX_RESULT_BYTES,
}


@contextmanager
def output_lock(directory):
    """Hold a Linux advisory lock for execution and report writes; never unlink it."""
    import fcntl

    Path(directory).mkdir(parents=True, exist_ok=True)
    fd = os.open(Path(directory) / ".evaluation.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluation owns this output directory") from exc
        yield
    finally:
        os.close(fd)


def extract_artifacts(archive, destination):
    """Extract into a fresh controller-owned directory, never following links.

    Validate the whole uncompressed archive before writing; discard permissions,
    owners, timestamps, and all archive metadata. No tarfile.extract calls.
    """
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Artifact destination must be new")
    with tarfile.open(fileobj=archive, mode="r:") as tar:
        members, seen, total = [], set(), 0
        for member in tar:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or not (member.isdir() or member.isfile()) or member.sparse is not None):
                raise ValueError(f"Unsafe artifact entry: {member.name!r}")
            if path == PurePosixPath(".") and member.isdir():
                continue
            if path in seen or path == PurePosixPath("."):
                raise ValueError("Duplicate or empty artifact name")
            seen.add(path)
            total += member.size
            if len(seen) > 10000 or total > MAX_ARCHIVE_BYTES or member.size < 0:
                raise ValueError("Artifact archive exceeds limits")
            members.append((member, path))
        files = {path for member, path in members if member.isfile()}
        if any(parent in files for _, path in members for parent in path.parents):
            raise ValueError("Artifact file/directory conflict")
        destination.mkdir(mode=0o700)
        for member, path in members:
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
            else:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with tar.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o600)


def failed_result(request, status, error, started):
    return {
        "query_id": request.query_id, "query": request.query, "workspace": "/workspace",
        "tool_calls": [], "result": "", "status": status, "error": error,
        "structured_output": {"_parse_error": error},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "usage": {"execution": None, "formatting": None,
                  "total": apply_history_coverage(summarize_usage([]), True)},
    }


class PodmanSetupError(RuntimeError):
    """Infrastructure/configuration errors abort the run instead of producing scores."""


class PodmanRuntime:
    def __init__(self, config):
        self.config = config
        self.run_id = uuid.uuid4().hex
        self.label = f"{LABEL}={self.run_id}"
        self.internal = f"cg-eval-{self.run_id}-internal"
        self.outbound = f"cg-eval-{self.run_id}-outbound"
        self.proxy = f"cg-eval-{self.run_id}-proxy"
        self.resources = []
        self.provenance = {}
        self.requests = {}
        self.env = {key: value for key, value in os.environ.items()
                    if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
                    and key not in {"CONTAINER_HOST", "CONTAINER_CONNECTION", "DOCKER_HOST"}}

    async def command(self, *args, data=None, timeout=30, output=None, limit=MAX_RESULT_BYTES):
        """Bound command output and duration, killing the client on cancellation."""
        proc = await asyncio.create_subprocess_exec(
            "podman", "--remote=false", *args, env=self.env,
            stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )

        async def drain(stream, sink=None):
            chunks, size = [], 0
            while chunk := await stream.read(65536):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Podman command output exceeds limit")
                if sink is None:
                    chunks.append(chunk)
                else:
                    sink.write(chunk)
            return b"".join(chunks)

        async def communicate():
            if data is not None:
                proc.stdin.write(data)
                await proc.stdin.drain()
                proc.stdin.close()
            stdout, stderr = await asyncio.gather(drain(proc.stdout, output), drain(proc.stderr))
            code = await proc.wait()
            if code:
                raise RuntimeError(f"Podman {args[0]} failed ({code}): {stderr.decode(errors='replace')[-4000:]}")
            return stdout

        try:
            return await asyncio.wait_for(communicate(), timeout)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    def security_args(self):
        return [
            "--pull=never", "--http-proxy=false", "--read-only", "--read-only-tmpfs=false",
            "--user=1000:1000", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=256", f"--cpus={self.config.podman_cpus}",
            f"--memory={self.memory_bytes}", f"--memory-swap={self.memory_bytes}",
            "--pid=private", "--ipc=private", "--uts=private", "--cgroupns=private",
            "--no-hosts", "--image-volume=ignore", "--log-driver=none",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=1073741824,mode=1777",
            "--tmpfs=/run:rw,nosuid,nodev,noexec,size=16777216,mode=755",
            "--label", self.label,
        ]

    @property
    def memory_bytes(self):
        return int(self.config.podman_memory_gb * 1024**3)

    def worker_args(self, name, volume):
        args = ["run", "--name", name, *self.security_args(), "--interactive",
                "--network", self.internal, "--volume", f"{volume}:/workspace:rw,U,Z",
                "--workdir=/workspace", "--entrypoint=python"]
        for key, value in {
            "HTTP_PROXY": f"http://{self.proxy}:3128", "HTTPS_PROXY": f"http://{self.proxy}:3128",
            "http_proxy": f"http://{self.proxy}:3128", "https_proxy": f"http://{self.proxy}:3128",
            "NO_PROXY": "", "no_proxy": "", "ALL_PROXY": "", "all_proxy": "",
            "HOME": "/tmp", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
            "CHEMGRAPH_LOG_DIR": "/workspace", "OMP_NUM_THREADS": str(max(1, int(self.config.podman_cpus))),
        }.items():
            args += ["--env", f"{key}={value}"]
        return args + [self.worker_image]

    async def image_id(self, image):
        info = json.loads(await self.command("image", "inspect", "--", image))[0]
        image_id = info["Id"]
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            raise ValueError("Podman did not return an immutable image ID")
        return image_id

    async def prepare(self):
        if platform.system() != "Linux" or not shutil.which("podman"):
            raise RuntimeError("Podman evaluation requires native Linux and installed rootless Podman")
        info = json.loads(await self.command("info", "--format=json"))
        host = info["host"]
        if (not host.get("security", {}).get("rootless") or host.get("networkBackend") != "netavark"
                or host.get("cgroupVersion") != "v2"
                or not {"cpu", "memory", "pids"}.issubset(host.get("cgroupControllers", []))):
            raise RuntimeError("Rootless Podman, Netavark and delegated cgroup v2 cpu/memory/pids are required")
        from chemgraph.models.endpoints.argo import resolve_anthropic_base_url, resolve_openai_base_url
        from chemgraph.models.supported_models import supported_argo_models

        user = self.config.get_argo_user() or os.environ.get("ARGO_USER", "")
        for model in self.config.models:
            if model not in supported_argo_models:
                raise ValueError(f"Unsupported Argo model: {model}")
            resolve = resolve_anthropic_base_url if model.startswith("argo:claude-") else resolve_openai_base_url
            self.requests[model] = WorkerRequest(
                query="preflight", query_id="preflight", model=model,
                base_url=resolve(model, self.config.get_base_url(model)), argo_user=user,
                execution=ExecutionSettings.from_config(self.config),
            )
        self.worker_image = await self.image_id(self.config.podman_image)
        self.proxy_image = await self.image_id(self.config.podman_proxy_image)
        manifest_name = f"cg-eval-{self.run_id}-manifest"
        self.resources.append(("container", manifest_name))
        try:
            manifest = json.loads(await self.command(
                "run", "--rm", "--name", manifest_name, *self.security_args(),
                "--network=none", "--entrypoint=cat",
                self.worker_image, "/opt/chemgraph-evaluation-manifest.json",
            ))
        finally:
            await self.remove("container", manifest_name)
        if (manifest.get("version") != 1 or not manifest.get("dependencies")
                or not manifest.get("weights") or manifest.get("model") != "medium-mpa-0"):
            raise ValueError("Worker image lacks the required dependency/weight manifest")
        self.provenance = {
            "runtime": "podman", "worker_image_id": self.worker_image, "proxy_image_id": self.proxy_image,
            "worker_manifest": manifest, "isolation_policy": POLICY,
            "resources": {"cpus": self.config.podman_cpus, "memory_bytes": self.memory_bytes,
                          "pids": 256, "timeout_seconds": self.config.query_timeout_seconds},
            "execution_endpoints": {model: request.base_url for model, request in self.requests.items()},
            "podman_version": info.get("version"),
        }
        for name, internal in [(self.internal, True), (self.outbound, False)]:
            self.resources.append(("network", name))
            await self.command("network", "create", "--driver=bridge", "--label", self.label,
                               *(["--internal"] if internal else []), name)
            network = json.loads(await self.command("network", "inspect", name))[0]
            if network["driver"] != "bridge" or network["internal"] != internal:
                raise RuntimeError("Podman did not create the requested isolated network")
        self.resources.append(("container", self.proxy))
        await self.command(
            "run", "--detach", "--name", self.proxy, *self.security_args(),
            "--network", self.outbound, "--network", self.internal,
            self.proxy_image,
        )
        name, volume = await self.new_worker()
        try:
            result = await self.launch_worker(
                name, volume, ["-m", "chemgraph.eval.isolation",
                               str(self.config.podman_cpus), str(self.memory_bytes)], timeout=120,
            )
            if json.loads(result) != {"isolation": "passed"}:
                raise RuntimeError("Worker isolation probe did not pass")
        finally:
            await self.remove("container", name)
            await self.remove("volume", volume)

    async def launch_worker(self, name, volume, command, *, data=None, timeout=30):
        # Inspect the actual mounts and network before any agent code can run.
        args = self.worker_args(name, volume)
        args[0] = "create"
        try:
            await self.command(*args, *command)
            inspected = json.loads(await self.command("container", "inspect", name))[0]
            self.validate_container(inspected, volume)
        except Exception as exc:
            raise PodmanSetupError(f"Cannot prepare isolated worker: {exc}") from exc
        return await self.command("start", "--attach", "--interactive", name, data=data, timeout=timeout)

    def validate_container(self, inspected, volume):
        """Reject ambient host defaults such as mounts.conf bind mounts or port mappings."""
        host = inspected["HostConfig"]
        if (host.get("Privileged") or not host.get("ReadonlyRootfs") or host.get("PortBindings")
                or inspected["Config"].get("User") != "1000:1000"
                or set(inspected["NetworkSettings"]["Networks"]) != {self.internal}):
            raise RuntimeError("Podman did not enforce the requested worker isolation")
        workspace_mounts = 0
        for mount in inspected["Mounts"]:
            if (mount["Type"] == "volume" and mount["Destination"] == "/workspace"
                    and mount.get("Name") == volume):
                workspace_mounts += 1
            elif mount["Type"] == "tmpfs" and mount["Destination"] in {"/tmp", "/run", "/dev/shm"}:
                continue
            else:
                raise RuntimeError("Unexpected worker mount; remove ambient Podman mount configuration")
        if workspace_mounts != 1:
            raise RuntimeError("Worker does not have exactly one private workspace volume")

    async def new_worker(self):
        token = uuid.uuid4().hex
        name, volume = f"cg-eval-{self.run_id}-{token}", f"cg-eval-{self.run_id}-{token}-data"
        self.resources.append(("volume", volume))
        await self.command("volume", "create", "--label", self.label, volume)
        self.resources.append(("container", name))
        return name, volume

    async def remove(self, kind, name):
        # Both an unguessable name and a matching run label are required.
        records = json.loads(await self.command(
            kind, "ls", *(["--all"] if kind == "container" else []),
            "--filter", f"label={self.label}", "--format=json",
        ))
        owned = any(name in [row.get("Name"), row.get("name"), *(row.get("Names") or [])] for row in records)
        if owned:
            await self.command(kind, "rm", *(["--force"] if kind == "container" else []), name)
        self.resources.remove((kind, name))

    async def close(self):
        errors = []
        for kind, name in list(reversed(self.resources)):
            try:
                await self.remove(kind, name)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError(f"Cleanup failed for run {self.run_id}: {'; '.join(errors)}")

    async def run_query(self, model, query, query_id):
        request = self.requests[model].model_copy(update={"query": query, "query_id": query_id})
        started = time.monotonic()
        # Resource creation errors are configuration/runtime failures, not model scores.
        name, volume = await self.new_worker()
        destination = Path(self.config.output_dir).resolve() / "artifacts" / uuid.uuid4().hex
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = None
        try:
            try:
                await self.launch_worker(
                    name, volume, ["-m", "chemgraph.eval.worker"],
                    data=request.model_dump_json().encode(), timeout=self.config.query_timeout_seconds,
                )
            except PodmanSetupError:
                raise
            except TimeoutError:
                raw = failed_result(request, "timeout", "Query exceeded its time limit", started)
            except (RuntimeError, ValueError) as exc:
                raw = failed_result(request, "worker_error", str(exc), started)
            # Stop all processes before copying, including descendants of a crashed worker.
            await self.command("stop", "--ignore", "--time=0", name)
            try:
                with tempfile.TemporaryFile() as archive:
                    await self.command("volume", "export", volume, output=archive, limit=MAX_ARCHIVE_BYTES)
                    archive.seek(0)
                    extract_artifacts(archive, destination)
                if raw is None:
                    result_path = destination / RESULT_FILE
                    if result_path.stat().st_size > MAX_RESULT_BYTES:
                        raise ValueError("Worker result exceeds size limit")
                    raw = validate_response(result_path.read_bytes(), request)
                raw["artifact_location"] = str(destination)
            except (OSError, ValueError, tarfile.TarError, RuntimeError) as exc:
                if raw is None:
                    raw = failed_result(request, "invalid_response", str(exc), started)
                else:
                    raw["artifact_error"] = str(exc)
            return raw
        finally:
            await self.remove("container", name)
            await self.remove("volume", volume)
