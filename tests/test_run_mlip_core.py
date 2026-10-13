import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write
from pydantic import ValidationError

from chemgraph.schemas.mlip_input import (
    MLIPBatchInputSchema,
    MLIPInputSchema,
)
from chemgraph.tools import run_mlip_core as core


def _write_copper(path, distance=2.5):
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    write(path, atoms)
    return atoms


def _single_params(input_file, output_file, **overrides):
    values = {
        "input_structure_file": str(input_file),
        "output_results_file": str(output_file),
        "model": {"family": "mace", "checkpoint": "unused.model"},
    }
    values.update(overrides)
    return MLIPInputSchema(**values)


def _emt_context(counter=None):
    @contextmanager
    def calculator_context(_params):
        if counter is not None:
            counter["entered"] += 1
        yield (
            EMT(),
            {},
            {"backend": "ase", "device": "cpu", "dtype": "float64"},
            {"family": "mace", "checkpoint": "unused.model"},
        )

    return calculator_context


def test_run_mlip_ase_energy_writes_backend_neutral_result(tmp_path, monkeypatch):
    input_file = tmp_path / "copper.xyz"
    output_file = tmp_path / "energy.json"
    _write_copper(input_file)
    monkeypatch.setitem(core._ASE_CONTEXT_FACTORIES, "ase", _emt_context())

    result = core.run_mlip_core(_single_params(input_file, output_file))

    assert result["status"] == "success"
    assert result["potential_energy"] == result["single_point_energy"]
    assert result["single_point_energy"] is not None
    stored = json.loads(output_file.read_text())
    assert stored["success"] is True
    assert stored["simulation_input"]["driver"] == "energy"
    assert stored["calculator_info"]["requested"]["backend"] == "ase"
    assert stored["calculator_info"]["resolved"]["device"] == "cpu"
    assert stored["model_info"]["requested"]["family"] == "mace"
    assert stored["potential_energy"] == stored["single_point_energy"]
    assert len(stored["forces"]) == 2


def test_run_mlip_ase_fixed_cell_optimization(tmp_path, monkeypatch):
    input_file = tmp_path / "copper.xyz"
    output_file = tmp_path / "opt.json"
    initial = _write_copper(input_file, distance=3.0)
    monkeypatch.setitem(core._ASE_CONTEXT_FACTORIES, "ase", _emt_context())

    params = _single_params(
        input_file,
        output_file,
        driver="opt",
        steps=2,
        fmax=0.01,
    )
    result = core.run_mlip_core(params)

    assert result["status"] == "success"
    stored = json.loads(output_file.read_text())
    assert stored["simulation_input"]["driver"] == "opt"
    assert stored["final_structure"]["cell"] == initial.cell.tolist()


def test_batch_reuses_calculator_and_records_partial_failure(tmp_path, monkeypatch):
    first = tmp_path / "first.xyz"
    second = tmp_path / "second.xyz"
    missing = tmp_path / "missing.xyz"
    _write_copper(first)
    _write_copper(second, distance=2.7)
    counter = {"entered": 0}
    monkeypatch.setitem(
        core._ASE_CONTEXT_FACTORIES, "ase", _emt_context(counter)
    )

    params = MLIPBatchInputSchema(
        input_structure_files=[str(first), str(missing), str(second)],
        output_results_directory=str(tmp_path / "results"),
        model={"family": "mace", "checkpoint": "unused.model"},
    )
    result = core.run_mlip_batch_core(params)

    assert result["status"] == "partial"
    assert result["succeeded"] == 2
    assert result["failed"] == 1
    assert counter["entered"] == 1
    manifest = json.loads((tmp_path / "results" / "batch_manifest.json").read_text())
    assert [item["index"] for item in manifest["items"]] == [0, 1, 2]
    assert [item["status"] for item in manifest["items"]] == [
        "success",
        "failure",
        "success",
    ]
    expected_results = [
        "00000_first.json",
        "00001_missing.json",
        "00002_second.json",
    ]
    assert all((tmp_path / "results" / name).exists() for name in expected_results)


def test_nvalchemi_batch_loads_once_and_chunks_in_order(tmp_path, monkeypatch):
    input_files = []
    for index in range(3):
        input_file = tmp_path / f"input_{index}.xyz"
        _write_copper(input_file, distance=2.5 + index * 0.1)
        input_files.append(str(input_file))

    loaded = []
    chunks = []
    sentinel_model = object()

    def load_model(params):
        loaded.append(params.model.checkpoint)
        return (
            sentinel_model,
            {"backend": "nvalchemi", "device": "cuda"},
            {"family": "mace", "checkpoint": "model.pt"},
        )

    def run_chunk(model, entries, calculator_resolved, model_resolved):
        assert model is sentinel_model
        assert calculator_resolved["backend"] == "nvalchemi"
        assert model_resolved["family"] == "mace"
        chunks.append([entry[1] for entry in entries])
        return [
            core._success_result(
                params,
                input_file,
                atoms,
                1.25,
                np.zeros((len(atoms), 3)),
                None,
                True,
                0.01,
            )
            for params, input_file, atoms, _ in entries
        ]

    monkeypatch.setattr(core, "_load_nvalchemi_model", load_model)
    monkeypatch.setattr(core, "_run_nvalchemi_chunk", run_chunk)
    params = MLIPBatchInputSchema(
        input_structure_files=input_files,
        output_results_directory=str(tmp_path / "results"),
        batch_size=2,
        calculator={"backend": "nvalchemi", "device": "cuda"},
        model={"family": "mace", "checkpoint": "model.pt"},
    )

    result = core.run_mlip_batch_core(params)

    assert result["status"] == "completed"
    assert loaded == ["model.pt"]
    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert chunks[0] + chunks[1] == input_files


def test_rootstock_calculator_uses_context_manager(monkeypatch):
    events = []

    class FakeRootstockCalculator:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def __enter__(self):
            events.append(("enter", None))
            return "calculator"

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", None))

    monkeypatch.setitem(
        sys.modules,
        "rootstock",
        SimpleNamespace(RootstockCalculator=FakeRootstockCalculator),
    )
    params = MLIPInputSchema(
        input_structure_file="unused.xyz",
        model={
            "family": "mace",
            "checkpoint": "org/model",
        },
        calculator={
            "backend": "rootstock",
            "cluster": "local",
        },
    )

    with core._rootstock_calculator_context(params) as (
        calculator,
        atoms_info,
        calculator_resolved,
        model_resolved,
    ):
        assert calculator == "calculator"
        assert atoms_info == {}
        assert calculator_resolved["backend"] == "rootstock"
        assert model_resolved == {"family": "mace", "checkpoint": "org/model"}

    assert [event[0] for event in events] == ["init", "enter", "exit"]
    assert events[0][1]["checkpoint"] == "org/model"
    assert events[0][1]["cluster"] == "local"


def test_optional_backend_error_is_persisted_per_item(tmp_path, monkeypatch):
    input_file = tmp_path / "input.xyz"
    output_file = tmp_path / "result.json"
    _write_copper(input_file)

    def missing_backend(_params):
        raise ImportError("Install ChemGraph with the 'nvalchemi_mace' extra.")

    monkeypatch.setattr(core, "_load_nvalchemi_model", missing_backend)
    params = _single_params(
        input_file,
        output_file,
        calculator={"backend": "nvalchemi"},
    )

    result = core.run_mlip_core(params)

    assert result["status"] == "failure"
    stored = json.loads(output_file.read_text())
    assert stored["success"] is False
    assert "nvalchemi_mace" in stored["error"]


def test_schema_rejects_unsupported_calculator_model_pairs():
    with pytest.raises(ValidationError, match="supports only model.family='mace'"):
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={"backend": "nvalchemi"},
            model={"family": "uma", "checkpoint": "uma-s-1p1"},
        )

    with pytest.raises(ValidationError, match="only MACE-MP"):
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={"backend": "nvalchemi"},
            model={
                "family": "mace",
                "checkpoint": "model.pt",
                "calculator_type": "mace_off",
            },
        )

    with pytest.raises(ValidationError, match="does not support MACE dispersion"):
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={"backend": "nvalchemi"},
            model={
                "family": "mace",
                "checkpoint": "model.pt",
                "dispersion": {},
            },
        )

    with pytest.raises(ValidationError, match="omit calculator_type"):
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={"backend": "rootstock"},
            model={
                "family": "mace",
                "checkpoint": "org/model",
                "calculator_type": "mace_mp",
            },
        )

    with pytest.raises(ValidationError, match="setup_kwargs"):
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={"backend": "rootstock"},
            model={
                "family": "uma",
                "checkpoint": "org/model",
                "task_name": "omol",
            },
        )

    with pytest.raises(ValidationError, match="both cluster and root"):
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={
                "backend": "rootstock",
                "cluster": "local",
                "root": "/remote/root",
            },
            model={"family": "aimnet2", "checkpoint": "org/model"},
        )


def test_schema_routes_one_config_to_calculator_adapters():
    requests = [
        MLIPInputSchema(
            input_structure_file="input.xyz",
            model={"family": "uma", "checkpoint": "uma-s-1p1"},
        ),
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={"backend": "nvalchemi"},
            model={"family": "mace", "checkpoint": "model.pt"},
        ),
        MLIPInputSchema(
            input_structure_file="input.xyz",
            calculator={"backend": "rootstock", "cluster": "local"},
            model={"family": "aimnet2", "checkpoint": "org/model"},
        ),
    ]

    assert [request.calculator.backend for request in requests] == [
        "ase",
        "nvalchemi",
        "rootstock",
    ]
    assert [request.model.family for request in requests] == [
        "uma",
        "mace",
        "aimnet2",
    ]


def test_relative_paths_resolve_through_chemgraph_log_dir(tmp_path, monkeypatch):
    input_file = tmp_path / "input.xyz"
    _write_copper(input_file)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    monkeypatch.setitem(core._ASE_CONTEXT_FACTORIES, "ase", _emt_context())

    result = core.run_mlip_core(
        _single_params("input.xyz", "nested/result.json")
    )

    assert result["status"] == "success"
    assert (tmp_path / "nested" / "result.json").is_file()
