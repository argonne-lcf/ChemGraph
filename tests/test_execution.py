"""Tests for the chemgraph.execution abstraction layer.

Tests cover:
- TaskSpec validation
- LocalBackend: python and shell tasks
- GlobusComputeBackend: python and shell tasks (mocked SDK)
- Backend factory (get_backend)
- Shared utilities: resolve_structure_files, gather_futures, write_results_jsonl
"""

import asyncio
import json
import os
import sys
import tempfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chemgraph.execution.base import ExecutionBackend, TaskSpec
from chemgraph.execution.local_backend import LocalBackend
from chemgraph.execution.utils import (
    gather_futures,
    make_per_structure_output,
    resolve_structure_files,
    write_results_jsonl,
)


# ── TaskSpec tests ──────────────────────────────────────────────────────


class TestTaskSpec:
    def test_python_task_minimal(self):
        spec = TaskSpec(task_id="t1", task_type="python", callable=abs, args=(42,))
        assert spec.task_id == "t1"
        assert spec.task_type == "python"
        assert spec.callable is abs
        assert spec.args == (42,)

    def test_shell_task_minimal(self):
        spec = TaskSpec(task_id="t2", task_type="shell", command="echo hello")
        assert spec.task_type == "shell"
        assert spec.command == "echo hello"

    def test_defaults(self):
        spec = TaskSpec(task_id="t3")
        assert spec.task_type == "python"
        assert spec.callable is None
        assert spec.args == ()
        assert spec.kwargs == {}
        assert spec.num_nodes == 1
        assert spec.processes_per_node == 1
        assert spec.gpus_per_task == 0


# ── LocalBackend tests ──────────────────────────────────────────────────


def _square(x):
    return x * x


def _add(a, b):
    return a + b


def _failing_fn():
    raise ValueError("intentional test error")


class TestLocalBackend:
    def test_python_task(self):
        backend = LocalBackend()
        backend.initialize(system="local", max_workers=2)
        try:
            task = TaskSpec(
                task_id="sq",
                task_type="python",
                callable=_square,
                args=(7,),
            )
            fut = backend.submit(task)
            assert isinstance(fut, Future)
            assert fut.result(timeout=10) == 49
        finally:
            backend.shutdown()

    def test_python_task_kwargs(self):
        backend = LocalBackend()
        backend.initialize(system="local", max_workers=2)
        try:
            task = TaskSpec(
                task_id="add",
                task_type="python",
                callable=_add,
                kwargs={"a": 3, "b": 5},
            )
            assert backend.submit(task).result(timeout=10) == 8
        finally:
            backend.shutdown()

    def test_shell_task(self):
        backend = LocalBackend()
        backend.initialize(system="local", max_workers=1)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                stdout_path = f.name

            task = TaskSpec(
                task_id="echo",
                task_type="shell",
                command="echo hello_world",
                stdout=stdout_path,
            )
            fut = backend.submit(task)
            fut.result(timeout=10)

            with open(stdout_path) as f:
                assert "hello_world" in f.read()
        finally:
            backend.shutdown()
            os.unlink(stdout_path)

    def test_submit_batch(self):
        backend = LocalBackend()
        backend.initialize(system="local", max_workers=4)
        try:
            tasks = [
                TaskSpec(
                    task_id=f"sq_{i}",
                    task_type="python",
                    callable=_square,
                    args=(i,),
                )
                for i in range(5)
            ]
            futures = backend.submit_batch(tasks)
            assert len(futures) == 5
            results = [f.result(timeout=10) for f in futures]
            assert results == [0, 1, 4, 9, 16]
        finally:
            backend.shutdown()

    def test_failing_task(self):
        backend = LocalBackend()
        backend.initialize(system="local", max_workers=1)
        try:
            task = TaskSpec(
                task_id="fail",
                task_type="python",
                callable=_failing_fn,
            )
            fut = backend.submit(task)
            with pytest.raises(ValueError, match="intentional test error"):
                fut.result(timeout=10)
        finally:
            backend.shutdown()

    def test_context_manager(self):
        with LocalBackend() as backend:
            backend.initialize(system="local", max_workers=1)
            task = TaskSpec(
                task_id="ctx",
                task_type="python",
                callable=_square,
                args=(3,),
            )
            assert backend.submit(task).result(timeout=10) == 9

    def test_not_initialized_raises(self):
        backend = LocalBackend()
        task = TaskSpec(task_id="x", callable=_square, args=(1,))
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.submit(task)

    def test_python_task_missing_callable(self):
        backend = LocalBackend()
        backend.initialize(system="local", max_workers=1)
        try:
            task = TaskSpec(task_id="no_fn", task_type="python")
            with pytest.raises(ValueError, match="requires a callable"):
                backend.submit(task)
        finally:
            backend.shutdown()

    def test_shell_task_missing_command(self):
        backend = LocalBackend()
        backend.initialize(system="local", max_workers=1)
        try:
            task = TaskSpec(task_id="no_cmd", task_type="shell")
            with pytest.raises(ValueError, match="requires a command"):
                backend.submit(task)
        finally:
            backend.shutdown()


# ── GlobusComputeBackend tests ──────────────────────────────────────────


def _make_mock_gc_modules():
    """Create mock globus_compute_sdk module and its classes."""
    mock_sdk = MagicMock()

    # Mock Executor: instances track submit calls and return Futures
    mock_executor_instance = MagicMock()
    mock_future = Future()
    mock_future.set_result(42)
    mock_executor_instance.submit.return_value = mock_future
    mock_sdk.Executor.return_value = mock_executor_instance

    # Mock ShellFunction
    mock_shell_fn_instance = MagicMock()
    mock_sdk.ShellFunction.return_value = mock_shell_fn_instance

    return mock_sdk, mock_executor_instance


class TestGlobusComputeBackend:
    def _patch_and_import(self, mock_sdk):
        """Patch globus_compute_sdk into sys.modules and import the backend."""
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            # Force re-import to pick up the mock
            import importlib

            import chemgraph.execution.globus_compute_backend as gc_mod

            importlib.reload(gc_mod)
            return gc_mod.GlobusComputeBackend

    def test_initialize_success(self):
        mock_sdk, mock_executor = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(system="polaris", endpoint_id="test-uuid-1234")

            assert backend._initialized is True
            mock_sdk.Executor.assert_called_once_with(endpoint_id="test-uuid-1234")

    def test_initialize_with_amqp_port(self):
        mock_sdk, mock_executor = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(
                system="polaris",
                endpoint_id="test-uuid",
                amqp_port=443,
            )

            mock_sdk.Executor.assert_called_once_with(
                endpoint_id="test-uuid", amqp_port=443
            )

    def test_initialize_missing_endpoint_id(self):
        mock_sdk, _ = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            with pytest.raises(ValueError, match="endpoint_id"):
                backend.initialize(system="polaris")

    def test_initialize_empty_endpoint_id(self):
        mock_sdk, _ = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            with pytest.raises(ValueError, match="endpoint_id"):
                backend.initialize(system="polaris", endpoint_id="")

    def test_initialize_import_error(self):
        """Verify helpful error when globus-compute-sdk is not installed."""
        with patch.dict(sys.modules, {"globus_compute_sdk": None}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            with pytest.raises(ImportError, match="globus-compute-sdk"):
                backend.initialize(endpoint_id="test-uuid")

    def test_submit_python_task(self):
        mock_sdk, mock_executor = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(endpoint_id="test-uuid")

            task = TaskSpec(
                task_id="py1",
                task_type="python",
                callable=_square,
                args=(7,),
            )
            fut = backend.submit(task)

            assert isinstance(fut, Future)
            mock_executor.submit.assert_called_once_with(_square, 7)

    def test_submit_python_task_with_kwargs(self):
        mock_sdk, mock_executor = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(endpoint_id="test-uuid")

            task = TaskSpec(
                task_id="py2",
                task_type="python",
                callable=_add,
                args=(3,),
                kwargs={"b": 5},
            )
            backend.submit(task)

            mock_executor.submit.assert_called_once_with(_add, 3, b=5)

    def test_submit_shell_task(self):
        mock_sdk, mock_executor = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(endpoint_id="test-uuid")

            task = TaskSpec(
                task_id="sh1",
                task_type="shell",
                command="echo hello",
            )
            backend.submit(task)

            # ShellFunction should be constructed with the command
            mock_sdk.ShellFunction.assert_called_once_with("echo hello")
            # And then submitted via the executor
            shell_fn_instance = mock_sdk.ShellFunction.return_value
            mock_executor.submit.assert_called_once_with(shell_fn_instance)

    def test_submit_not_initialized(self):
        from chemgraph.execution.globus_compute_backend import (
            GlobusComputeBackend,
        )

        backend = GlobusComputeBackend()
        task = TaskSpec(task_id="x", callable=_square, args=(1,))
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.submit(task)

    def test_submit_python_missing_callable(self):
        mock_sdk, _ = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(endpoint_id="test-uuid")

            task = TaskSpec(task_id="no_fn", task_type="python")
            with pytest.raises(ValueError, match="requires a callable"):
                backend.submit(task)

    def test_submit_shell_missing_command(self):
        mock_sdk, _ = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(endpoint_id="test-uuid")

            task = TaskSpec(task_id="no_cmd", task_type="shell")
            with pytest.raises(ValueError, match="requires a command"):
                backend.submit(task)

    def test_shutdown(self):
        mock_sdk, mock_executor = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = GlobusComputeBackend()
            backend.initialize(endpoint_id="test-uuid")
            assert backend._initialized is True

            backend.shutdown()

            assert backend._initialized is False
            assert backend._executor is None
            mock_executor.shutdown.assert_called_once()

    def test_shutdown_idempotent(self):
        """Calling shutdown() when not initialized should not raise."""
        from chemgraph.execution.globus_compute_backend import (
            GlobusComputeBackend,
        )

        backend = GlobusComputeBackend()
        backend.shutdown()  # should be a no-op
        assert backend._initialized is False

    def test_context_manager(self):
        mock_sdk, mock_executor = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            with GlobusComputeBackend() as backend:
                backend.initialize(endpoint_id="test-uuid")
                task = TaskSpec(
                    task_id="ctx",
                    task_type="python",
                    callable=_square,
                    args=(3,),
                )
                backend.submit(task)

            # After exiting context, shutdown should have been called
            mock_executor.shutdown.assert_called_once()


class TestGetBackendGlobusCompute:
    def test_factory_creates_globus_compute_backend(self):
        mock_sdk, _ = _make_mock_gc_modules()
        with patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}):
            from chemgraph.execution.config import get_backend
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = get_backend(
                backend_name="globus_compute",
                endpoint_id="factory-test-uuid",
            )
            try:
                assert isinstance(backend, GlobusComputeBackend)
                assert backend._initialized is True
            finally:
                backend.shutdown()

    def test_factory_via_env_var(self):
        mock_sdk, _ = _make_mock_gc_modules()
        with (
            patch.dict(sys.modules, {"globus_compute_sdk": mock_sdk}),
            patch.dict(
                os.environ,
                {"CHEMGRAPH_EXECUTION_BACKEND": "globus_compute"},
            ),
        ):
            from chemgraph.execution.config import get_backend
            from chemgraph.execution.globus_compute_backend import (
                GlobusComputeBackend,
            )

            backend = get_backend(endpoint_id="env-test-uuid")
            try:
                assert isinstance(backend, GlobusComputeBackend)
            finally:
                backend.shutdown()


# ── Factory tests ───────────────────────────────────────────────────────


class TestGetBackend:
    def test_local_backend_via_env(self):
        with patch.dict(os.environ, {"CHEMGRAPH_EXECUTION_BACKEND": "local"}):
            from chemgraph.execution.config import get_backend

            backend = get_backend()
            try:
                assert isinstance(backend, LocalBackend)
            finally:
                backend.shutdown()

    def test_explicit_backend_name(self):
        from chemgraph.execution.config import get_backend

        backend = get_backend(backend_name="local", max_workers=2)
        try:
            assert isinstance(backend, LocalBackend)
        finally:
            backend.shutdown()

    def test_unsupported_backend_raises(self):
        from chemgraph.execution.config import get_backend

        with pytest.raises(ValueError, match="Unknown execution backend"):
            get_backend(backend_name="nonexistent")


# ── Utility tests ───────────────────────────────────────────────────────


class TestResolveStructureFiles:
    def test_from_directory(self, tmp_path):
        for name in ["a.cif", "b.cif", "c.txt"]:
            (tmp_path / name).write_text("dummy")

        files, out_dir = resolve_structure_files(str(tmp_path), extensions={".cif"})
        assert len(files) == 2
        assert out_dir == tmp_path
        assert all(f.suffix == ".cif" for f in files)

    def test_from_file_list(self, tmp_path):
        paths = []
        for name in ["x.xyz", "y.xyz"]:
            p = tmp_path / name
            p.write_text("dummy")
            paths.append(str(p))

        files, out_dir = resolve_structure_files(paths)
        assert len(files) == 2
        assert out_dir == tmp_path

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            resolve_structure_files([str(tmp_path / "noexist.cif")])

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No structure files"):
            resolve_structure_files(str(tmp_path), extensions={".cif"})

    def test_invalid_dir_raises(self):
        with pytest.raises(ValueError, match="not a valid directory"):
            resolve_structure_files("/nonexistent/path")


class TestMakePerStructureOutput:
    def test_basic(self):
        result = make_per_structure_output(
            Path("/data/MOF-5.cif"),
            Path("/results/output.json"),
        )
        assert result == Path("/results/MOF-5_output.json")

    def test_no_suffix(self):
        result = make_per_structure_output(
            Path("/data/struct.xyz"),
            Path("/results/result"),
        )
        assert result == Path("/results/struct_result.json")


class TestGatherFutures:
    @pytest.mark.asyncio
    async def test_successful_futures(self):
        loop = asyncio.get_event_loop()

        def _make_resolved(val):
            f = Future()
            f.set_result(val)
            return f

        pending = [
            ({"name": "a"}, _make_resolved({"status": "success", "energy": -1.0})),
            ({"name": "b"}, _make_resolved({"status": "success", "energy": -2.0})),
        ]
        results = await gather_futures(pending)
        assert len(results) == 2
        assert results[0]["name"] == "a"
        assert results[0]["energy"] == -1.0

    @pytest.mark.asyncio
    async def test_failed_future(self):
        f = Future()
        f.set_exception(RuntimeError("boom"))

        pending = [({"name": "fail"}, f)]
        results = await gather_futures(pending)
        assert results[0]["status"] == "failure"
        assert results[0]["error_type"] == "RuntimeError"
        assert "boom" in results[0]["message"]

    @pytest.mark.asyncio
    async def test_with_post_fn(self):
        f = Future()
        f.set_result(42)

        def post(meta, result):
            return {**meta, "doubled": result * 2, "status": "success"}

        results = await gather_futures([({"id": "x"}, f)], post_fn=post)
        assert results[0]["doubled"] == 84


class TestWriteResultsJsonl:
    def test_write_and_count(self, tmp_path):
        results = [
            {"status": "success", "value": 1},
            {"status": "failure", "error": "bad"},
            {"status": "success", "value": 2},
        ]
        path = tmp_path / "results.jsonl"
        success, total = write_results_jsonl(results, path)
        assert success == 2
        assert total == 3

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["value"] == 1

    def test_append_mode(self, tmp_path):
        path = tmp_path / "results.jsonl"
        write_results_jsonl([{"status": "success"}], path)
        write_results_jsonl([{"status": "success"}], path, append=True)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
