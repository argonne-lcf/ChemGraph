from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from chemgraph.execution.scoped import execution_context
from chemgraph.schemas.xanes_schema import xanes_input_schema
from chemgraph.tools import xanes_core
from tests.test_scoped_execution import scope


@pytest.mark.parametrize(
    "escape",
    [
        "default-directory",
        "explicit-directory",
        "fdmfile.txt",
        "fdmnes_in.txt",
        "Z26_local_Fe.pkl",
        "fdmnes_stdout.txt",
        "fdmnes_stderr.txt",
    ],
)
def test_xanes_rejects_output_symlinks_before_writing(tmp_path, monkeypatch, escape):
    context = scope(tmp_path)
    (context.workspace / "fe.xyz").write_text("1\nIron\nFe 0 0 0\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("unchanged")
    run = context.workspace / "fdmnes_fe"
    if escape.endswith("directory"):
        run.symlink_to(outside, target_is_directory=True)
    else:
        run.mkdir()
        (run / escape).symlink_to(sentinel)
    engine = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setenv("FDMNES_EXE", "fake-engine")
    monkeypatch.setattr(xanes_core.subprocess, "run", engine)
    params = xanes_input_schema(
        input_structure_file="fe.xyz",
        output_dir="fdmnes_fe" if escape == "explicit-directory" else None,
    )
    with execution_context(context), pytest.raises(PermissionError):
        xanes_core.run_xanes_core(params)
    engine.assert_not_called()
    assert sentinel.read_text() == "unchanged"
    assert list(outside.iterdir()) == [sentinel]


def test_xanes_writes_normal_scoped_outputs(tmp_path, monkeypatch):
    context = scope(tmp_path)
    (context.workspace / "fe.xyz").write_text("1\nIron\nFe 0 0 0\n")
    engine = Mock(return_value=SimpleNamespace(returncode=1))
    monkeypatch.setenv("FDMNES_EXE", "fake-engine")
    monkeypatch.setattr(xanes_core.subprocess, "run", engine)
    with execution_context(context):
        result = xanes_core.run_xanes_core(
            xanes_input_schema(input_structure_file="fe.xyz")
        )
    assert result["output_dir"] == str(context.workspace / "fdmnes_fe")
    engine.assert_called_once()
    assert len(list((context.workspace / "fdmnes_fe").iterdir())) == 5
