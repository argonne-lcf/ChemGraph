"""Offline boundary, failure, and provenance tests for optional Podman execution."""

import asyncio
import io
import json
from pathlib import Path
import sys
import tarfile
import time
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from chemgraph.agent.usage import summarize_usage
from chemgraph.eval.cli import build_config_from_args, parse_args
from chemgraph.eval.config import BenchmarkConfig
from chemgraph.eval.podman import PodmanRuntime, extract_artifacts, failed_result, output_lock
from chemgraph.eval.protocol import ExecutionSettings, WorkerRequest, validate_response
from chemgraph.eval.runner import ModelBenchmarkRunner


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Podman evaluation targets native Linux")


@pytest.fixture
def config(tmp_path):
    dataset = tmp_path / "answers.json"
    dataset.write_text(json.dumps([{
        "id": "3", "query": "Find water", "answer": {"result": "HOST_ONLY_ANSWER",
                                                        "structured_output": {"smiles": ["O"]}},
    }]))
    return BenchmarkConfig(
        models=["argo:gpt-5.6-sol"], workflow_types=["deep_agent"], dataset=str(dataset),
        output_dir=str(tmp_path / "results"), judge_type="structured", deepagent_auto_approve=True,
        eval_runtime="podman", podman_image="worker", podman_proxy_image="proxy",
    )


@pytest.fixture
def request_data():
    return dict(query="Find water", query_id="3", model="argo:gpt-5.6-sol",
                base_url="https://apps.inside.anl.gov/argoapi/v1", argo_user="test",
                execution=ExecutionSettings().model_dump())


def response(request):
    return {"version": 1, "raw": {
        "query_id": request.query_id, "query": request.query, "workspace": "/workspace",
        "tool_calls": [], "result": "Water", "structured_output": {"smiles": ["O"]},
        "status": "completed", "elapsed_seconds": 1.0,
        "usage": {"execution": None, "formatting": None, "total": summarize_usage([])},
    }}


@pytest.mark.parametrize("key", ["dataset", "expected_result", "output_dir", "config", "config_file", "skills", "workspace"])
def test_request_rejects_host_configuration(request_data, key):
    with pytest.raises(ValidationError):
        WorkerRequest(**request_data, **{key: "HOST_ONLY_ANSWER"})
    with pytest.raises(ValidationError):
        WorkerRequest(**{**request_data, "execution": {**request_data["execution"], key: "host"}})


@pytest.mark.parametrize("url", [
    "http://apps.inside.anl.gov/argoapi/v1", "https://apps.inside.anl.gov.evil/argoapi/v1",
    "https://127.0.0.1/argoapi/v1", "https://apps.inside.anl.gov:444/argoapi/v1",
    "https://user@apps.inside.anl.gov/argoapi/v1", "https://apps.inside.anl.gov/argoapi/v1?token=x",
])
def test_request_requires_hosted_argo(request_data, url):
    with pytest.raises(ValidationError):
        WorkerRequest(**{**request_data, "base_url": url})


@pytest.mark.parametrize("change", [
    {"eval_runtime": "docker"}, {"podman_cpus": 0}, {"podman_memory_gb": float("inf")},
    {"query_timeout_seconds": -1}, {"podman_image": None}, {"podman_proxy_image": None},
    {"workflow_types": ["single_agent"]}, {"models": ["gpt-4o"]}, {"deepagent_workspace": "/tmp"},
])
def test_invalid_runtime_config(config, change):
    with pytest.raises(ValidationError):
        BenchmarkConfig(**{**config.model_dump(), **change})


def test_profile_and_cli(config, tmp_path):
    profile = tmp_path / "profile.toml"
    profile.write_text('[eval.profiles.isolated]\n'
                       f'dataset = {json.dumps(config.dataset)}\n'
                       'workflow_types = ["deep_agent"]\ndeepagent_auto_approve = true\n'
                       'judge_type = "structured"\neval_runtime = "podman"\n'
                       'podman_image = "worker"\npodman_proxy_image = "proxy"\n'
                       'podman_cpus = 3\npodman_memory_gb = 5\nquery_timeout_seconds = 19\n')
    args = ["--models", *config.models, "--profile", "isolated", "--config", str(profile)]
    resolved = build_config_from_args(parse_args(args))
    assert (resolved.podman_cpus, resolved.podman_memory_gb, resolved.query_timeout_seconds) == (3, 5, 19)
    resolved = build_config_from_args(parse_args(args + ["--podman-cpus", "2", "--query-timeout-seconds", "10"]))
    assert (resolved.podman_cpus, resolved.query_timeout_seconds) == (2, 10)
    assert resolved.deepagent_workspace is None


def test_container_arguments_have_no_host_mounts(config, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-host-proxy:123")
    runtime = PodmanRuntime(config)
    runtime.worker_image = "sha256:" + "a" * 64
    args = runtime.worker_args("worker", "private-volume")
    assert args.count("--volume") == 1
    assert args[args.index("--volume") + 1] == "private-volume:/workspace:rw,U,Z"
    assert args[args.index("--network") + 1] == runtime.internal
    assert "--http-proxy=false" in args and "--pull=never" in args
    for flag in ("--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                 "--pids-limit=256", "--no-hosts", "--user=1000:1000", "--image-volume=ignore"):
        assert flag in args
    serialized = json.dumps(args)
    assert config.dataset not in serialized and config.output_dir not in serialized
    assert "secret-host-proxy" not in serialized
    assert "HTTPS_PROXY" not in runtime.env


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(version=2),
    lambda r: r["raw"].update(query_id="other"),
    lambda r: r["raw"].update(score=1),
    lambda r: r["raw"].update(status="timeout"),
    lambda r: r["raw"].update(status="execution_error"),
    lambda r: r["raw"].update(error="failure"),
    lambda r: r["raw"].update(workspace="/home/user"),
    lambda r: r["raw"].update(elapsed_seconds=float("nan")),
    lambda r: r["raw"].update(structured_output=None),
    lambda r: r["raw"].update(usage={"total": {}}),
    lambda r: r["raw"]["usage"]["total"].update(call_count=-1),
])
def test_invalid_results(request_data, mutation):
    request = WorkerRequest(**request_data)
    result = response(request)
    mutation(result)
    with pytest.raises(ValueError):
        validate_response(json.dumps(result).encode(), request)


def archive_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, kind, content in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(content) if kind == tarfile.REGTYPE else 0
            member.linkname = "/etc/passwd" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
            archive.addfile(member, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    stream.seek(0)
    return stream


@pytest.mark.parametrize("name,kind", [
    ("../answers.json", tarfile.REGTYPE), ("/tmp/overwrite", tarfile.REGTYPE),
    ("a/../../overwrite", tarfile.REGTYPE), ("a\\..\\overwrite", tarfile.REGTYPE),
    ("link", tarfile.SYMTYPE), ("hard", tarfile.LNKTYPE), ("device", tarfile.CHRTYPE),
    ("fifo", tarfile.FIFOTYPE),
])
def test_artifact_attacks_are_rejected_before_writing(tmp_path, name, kind):
    target = tmp_path / "artifacts"
    source = archive_bytes([("safe", tarfile.REGTYPE, b"ok"), (name, kind, b"bad")])
    with pytest.raises(ValueError):
        extract_artifacts(source, target)
    assert not target.exists()


@pytest.mark.parametrize("names", [["same", "same"], ["file", "file/child"]])
def test_archive_name_conflicts(tmp_path, names):
    with pytest.raises(ValueError):
        extract_artifacts(archive_bytes([(n, tarfile.REGTYPE, b"x") for n in names]), tmp_path / "new")


def test_archive_success_and_lock(tmp_path):
    target = tmp_path / "artifacts"
    extract_artifacts(archive_bytes([("./a/b.txt", tarfile.REGTYPE, b"data")]), target)
    assert (target / "a/b.txt").read_bytes() == b"data"
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another evaluation"):
            with output_lock(tmp_path):
                pass
    with output_lock(tmp_path):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "crash", "invalid", "cancel"])
async def test_query_failure_and_cleanup(config, request_data, monkeypatch, failure):
    runtime = PodmanRuntime(config)
    runtime.worker_image = "sha256:" + "a" * 64
    request = WorkerRequest(**request_data)
    runtime.requests[request.model] = request
    events = []

    async def command(*args, **kwargs):
        events.append(args)
        if args[0] == "start":
            if failure == "timeout":
                raise TimeoutError
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "crash":
                raise RuntimeError("worker crashed")
        if args[:2] == ("volume", "export"):
            kwargs["output"].write(archive_bytes([(".chemgraph-result.json", tarfile.REGTYPE, b"bad json")]).getvalue())
        return b""

    monkeypatch.setattr(runtime, "command", command)

    async def launch_worker(*args, **kwargs):
        return await command("start")

    monkeypatch.setattr(runtime, "launch_worker", launch_worker)
    remove = AsyncMock()
    monkeypatch.setattr(runtime, "remove", remove)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await runtime.run_query(request.model, request.query, request.query_id)
    else:
        raw = await runtime.run_query(request.model, request.query, request.query_id)
        assert raw["status"] == {"timeout": "timeout", "crash": "worker_error", "invalid": "invalid_response"}[failure]
        assert raw["structured_output"].get("_parse_error")
        assert next(i for i, a in enumerate(events) if a[0] == "stop") < next(
            i for i, a in enumerate(events) if a[:2] == ("volume", "export"))
    assert [call.args[0] for call in remove.await_args_list] == ["container", "volume"]


@pytest.mark.asyncio
async def test_run_scoring_resume_and_fingerprints(config, request_data, monkeypatch):
    request = WorkerRequest(**request_data)
    calls = []
    manifest = {"worker_image_id": "image-one", "proxy_image_id": "proxy-one", "isolation_policy": 1}

    async def prepare(runtime):
        runtime.provenance = manifest.copy()

    async def run_query(runtime, model, query, query_id):
        calls.append((model, query, query_id))
        return failed_result(request, "timeout", "timeout", time.monotonic())

    monkeypatch.setattr(PodmanRuntime, "prepare", prepare)
    monkeypatch.setattr(PodmanRuntime, "run_query", run_query)
    monkeypatch.setattr(PodmanRuntime, "close", AsyncMock())
    runner = ModelBenchmarkRunner(config)
    result = (await runner.run_all())[request.model]["deep_agent"]
    assert result["structured_judge_aggregate"]["n_correct"] == 0
    assert result["structured_judge_aggregate"]["n_queries"] == 1
    config.resume = True
    assert (await ModelBenchmarkRunner(config).run_all())[request.model]["deep_agent"] == result
    assert len(calls) == 1  # Completed failure records are reused.
    manifest["worker_image_id"] = "image-two"
    with pytest.raises(ValueError, match="Incompatible"):
        await ModelBenchmarkRunner(config).run_all()
    manifest["worker_image_id"] = "image-one"
    config.podman_cpus = 2
    with pytest.raises(ValueError, match="Incompatible"):
        await ModelBenchmarkRunner(config).run_all()


@pytest.mark.asyncio
async def test_missing_podman_aborts_before_queries(config, monkeypatch):
    monkeypatch.setattr("chemgraph.eval.podman.shutil.which", lambda name: None)
    run = AsyncMock()
    monkeypatch.setattr(PodmanRuntime, "run_query", run)
    with pytest.raises(RuntimeError, match="installed rootless Podman"):
        await ModelBenchmarkRunner(config).run_all()
    run.assert_not_called()
    assert not (Path(config.output_dir) / "checkpoints").exists()


def test_build_context_excludes_user_data(tmp_path):
    from evaluation.build import stage_context

    stage_context(tmp_path)
    assert (tmp_path / "src/chemgraph/eval/worker.py").exists()
    assert not (tmp_path / "config.toml").exists()
    assert not (tmp_path / "tests").exists()
    assert not list(tmp_path.rglob("*.zip"))
    assert not list(tmp_path.rglob("output_data.json"))
    assert not (tmp_path / "src/chemgraph/eval/data").exists()


@pytest.mark.asyncio
async def test_judge_exception_is_checkpointed(config, request_data, monkeypatch):
    request = WorkerRequest(**request_data)

    async def prepare(runtime):
        runtime.provenance = {"worker_image_id": "test-image"}

    monkeypatch.setattr(PodmanRuntime, "prepare", prepare)
    monkeypatch.setattr(PodmanRuntime, "run_query", AsyncMock(return_value=response(request)["raw"]))
    monkeypatch.setattr(PodmanRuntime, "close", AsyncMock())

    def broken_judge(**kwargs):
        raise ValueError("invalid structured answer")

    monkeypatch.setattr("chemgraph.eval.runner.judge_structured_output", broken_judge)
    result = (await ModelBenchmarkRunner(config).run_all())[request.model]["deep_agent"]
    assert result["structured_judge_aggregate"]["n_correct"] == 0
    detail = result["structured_judge_details"][0]
    assert "Structured judge failed" in detail["rationale"]
    assert list((Path(config.output_dir) / "checkpoints").glob("*.jsonl"))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["rootless", "network", "cgroups"])
async def test_startup_rejects_missing_isolation(config, monkeypatch, missing):
    monkeypatch.setattr("chemgraph.eval.podman.shutil.which", lambda name: "/usr/bin/podman")
    host = {"security": {"rootless": True}, "networkBackend": "netavark",
            "cgroupVersion": "v2", "cgroupControllers": ["cpu", "memory", "pids"]}
    if missing == "rootless":
        host["security"]["rootless"] = False
    elif missing == "network":
        host["networkBackend"] = "cni"
    else:
        host["cgroupControllers"] = ["pids"]
    runtime = PodmanRuntime(config)
    command = AsyncMock(return_value=json.dumps({"host": host}).encode())
    monkeypatch.setattr(runtime, "command", command)
    with pytest.raises(RuntimeError, match="Rootless Podman"):
        await runtime.prepare()
    assert command.await_count == 1
    assert runtime.resources == []


@pytest.mark.asyncio
async def test_startup_resolves_images_endpoints_and_probes(config, monkeypatch):
    monkeypatch.setattr("chemgraph.eval.podman.shutil.which", lambda name: "/usr/bin/podman")
    monkeypatch.setenv("ARGO_USER", "fixture-user")
    runtime = PodmanRuntime(config)
    events = []
    images = {"worker": "sha256:" + "a" * 64, "proxy": "sha256:" + "b" * 64}

    async def command(*args, **kwargs):
        events.append(args)
        if args[0] == "info":
            value = {"host": {"security": {"rootless": True}, "networkBackend": "netavark",
                             "cgroupVersion": "v2", "cgroupControllers": ["cpu", "memory", "pids"]}}
        elif args[:2] == ("image", "inspect"):
            value = [{"Id": images[args[-1]]}]
        elif "--entrypoint=cat" in args:
            value = {"version": 1, "dependencies": {"torch": "2.14.1+cpu"},
                     "weights": {"/opt/model-cache/model": "a" * 64}, "model": "medium-mpa-0"}
        elif args[:2] == ("network", "inspect"):
            value = [{"driver": "bridge", "internal": args[-1] == runtime.internal}]
        elif args[0] == "start":
            value = {"isolation": "passed"}
        elif args[:2] == ("container", "inspect"):
            value = [{"HostConfig": {"ReadonlyRootfs": True}, "Config": {"User": "1000:1000"},
                      "NetworkSettings": {"Networks": {runtime.internal: {}}},
                      "Mounts": [{"Type": "volume", "Destination": "/workspace", "Name": args[-1] + "-data"}]}]
        elif len(args) > 1 and args[1] == "ls":
            value = []
        else:
            return b""
        return json.dumps(value).encode()

    monkeypatch.setattr(runtime, "command", command)
    await runtime.prepare()
    assert runtime.provenance["worker_image_id"] == images["worker"]
    assert runtime.provenance["proxy_image_id"] == images["proxy"]
    request = runtime.requests[config.models[0]]
    serialized = request.model_dump_json()
    assert request.argo_user == "fixture-user" and request.base_url.startswith("https://apps.inside.anl.gov/")
    assert "HOST_ONLY_ANSWER" not in serialized and config.dataset not in serialized
    assert "podman" not in serialized and "output_dir" not in serialized
    assert any("chemgraph.eval.isolation" in args for args in events)
    await runtime.close()
    assert runtime.resources == []


@pytest.mark.asyncio
async def test_cleanup_requires_matching_label_and_name(config, monkeypatch):
    runtime = PodmanRuntime(config)
    runtime.resources.append(("container", "ours"))
    command = AsyncMock(return_value=b'[{"Names": ["someone-elses-container"]}]')
    monkeypatch.setattr(runtime, "command", command)
    await runtime.remove("container", "ours")
    assert command.await_count == 1  # Never remove another evaluation's container.
    assert f"label={runtime.label}" in command.await_args.args
    assert "--all" in command.await_args.args


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_podman_client_is_reaped_on_timeout_or_cancel(config, tmp_path, monkeypatch, cancel):
    import os
    import sys

    executable = tmp_path / "podman"
    pidfile = tmp_path / "pid"
    executable.write_text(f"#!{sys.executable}\nimport os, time\n"
                          f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    runtime = PodmanRuntime(config)
    task = asyncio.create_task(runtime.command("info", timeout=1 if not cancel else 30))
    if cancel:
        for _ in range(100):
            if pidfile.exists():
                break
            await asyncio.sleep(0.01)
        assert pidfile.exists()
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_ambient_host_mount_is_rejected(config):
    runtime = PodmanRuntime(config)
    inspected = {"HostConfig": {"ReadonlyRootfs": True}, "Config": {"User": "1000:1000"},
                 "NetworkSettings": {"Networks": {runtime.internal: {}}},
                 "Mounts": [{"Type": "volume", "Destination": "/workspace", "Name": "private"}]}
    runtime.validate_container(inspected, "private")
    inspected["Mounts"].append({"Type": "bind", "Destination": "/secrets", "Source": "/host/home"})
    with pytest.raises(RuntimeError, match="Unexpected worker mount"):
        runtime.validate_container(inspected, "private")


@pytest.mark.asyncio
async def test_late_infrastructure_failure_aborts_run(config, monkeypatch):
    monkeypatch.setattr(PodmanRuntime, "prepare", AsyncMock())
    monkeypatch.setattr(PodmanRuntime, "run_query", AsyncMock(side_effect=RuntimeError("volume creation failed")))
    close = AsyncMock()
    monkeypatch.setattr(PodmanRuntime, "close", close)
    with pytest.raises(RuntimeError, match="volume creation failed"):
        await ModelBenchmarkRunner(config).run_all()
    close.assert_awaited_once()
    assert not list(Path(config.output_dir).rglob("*.jsonl"))


@pytest.mark.asyncio
async def test_worker_is_inspected_before_stdin_delivery(config, monkeypatch):
    runtime = PodmanRuntime(config)
    runtime.worker_image = "sha256:" + "a" * 64
    events = []

    async def command(*args, **kwargs):
        events.append((args, kwargs))
        if args[:2] == ("container", "inspect"):
            return json.dumps([{
                "HostConfig": {"ReadonlyRootfs": True}, "Config": {"User": "1000:1000"},
                "NetworkSettings": {"Networks": {runtime.internal: {}}},
                "Mounts": [{"Type": "volume", "Destination": "/workspace", "Name": "private"}],
            }]).encode()
        return b""

    monkeypatch.setattr(runtime, "command", command)
    await runtime.launch_worker("worker", "private", ["-m", "chemgraph.eval.worker"], data=b"question")
    assert [args[0] for args, _ in events] == ["create", "container", "start"]
    assert all("data" not in kwargs for _, kwargs in events[:2])
    assert events[-1][1]["data"] == b"question"
