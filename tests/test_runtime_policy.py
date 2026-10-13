"""Exact preparation of ASE tool calls and workspace confinement."""

from pathlib import Path

import pytest

from chemgraph.runtime import (
    ExecutionPolicy, PolicyDenied, PreparationError, PreparedAction, run_context, scoped_path,
)
from chemgraph.runtime.policy import prepare, validate
from tests.runtime_helpers import make_context, run_ase_call


def test_preparation_snapshots_inputs_with_workspace_relative_paths(tmp_path):
    context = make_context(tmp_path)
    action = prepare(context, run_ase_call())
    params = action.arguments["params"]
    snapshot = action.inputs[0]
    assert snapshot["source"] == "h2.xyz"
    assert params["input_structure_file"] == snapshot["path"]
    assert snapshot["path"].startswith(".chemgraph-inputs/")
    assert params["output_results_file"] == "energy.json"
    assert params["calculator"]["calculator_type"] == "emt"
    assert action.calculator["package"] == "ase"
    assert PreparedAction.from_dict(action.to_dict()) == action
    # Changing the original file afterwards cannot change the prepared input.
    (context.workspace / "h2.xyz").write_text("changed")
    assert (context.workspace / snapshot["path"]).read_text().startswith("2\n")
    validate(context, run_ase_call(), action)


def test_digest_covers_inputs_and_rejects_tampering(tmp_path):
    context = make_context(tmp_path)
    first = prepare(context, run_ase_call())
    (context.workspace / "h2.xyz").write_text("2\nEMT test\nH 0 0 0\nH 0 0 0.8\n")
    assert prepare(context, run_ase_call()).digest != first.digest
    tampered = {**first.to_dict(), "arguments": {"params": {}}}
    with pytest.raises(ValueError, match="digest"):
        PreparedAction.from_dict(tampered)
    (context.workspace / first.inputs[0]["path"]).write_text("tampered")
    with pytest.raises(PreparationError, match="input changed"):
        validate(context, run_ase_call(), first)


@pytest.mark.parametrize("call, error, match", [
    ({"name": "run_python", "args": {}, "id": "x"}, PolicyDenied, "not permitted"),
    (run_ase_call(structure="missing.xyz"), PreparationError, "does not exist yet"),
    (run_ase_call(structure="../outside.xyz"), PolicyDenied, "outside"),
    (run_ase_call(output="/tmp/energy.json"), PolicyDenied, "outside"),
    (run_ase_call(output="energy.txt"), PreparationError, ".json"),
    (run_ase_call(driver="md"), PreparationError, "driver"),
    (run_ase_call(steps=10_000), PolicyDenied, "step limit"),
    (run_ase_call(structure=".chemgraph-inputs/x/h2.xyz"), PolicyDenied, "Internal"),
])
def test_unpreparable_calls_fail_before_side_effects(tmp_path, call, error, match):
    context = make_context(tmp_path, limits={"max_steps": 5000})
    (tmp_path / "workspace" / "outside.xyz").write_text("2\n\nH 0 0 0\nH 0 0 1\n")
    with pytest.raises(error, match=match):
        prepare(context, call)


def test_calculator_selection_is_explicit_and_permitted(tmp_path):
    context = make_context(tmp_path)
    call = run_ase_call()
    call["args"]["params"].pop("calculator")
    with pytest.raises(PreparationError, match="explicitly"):
        prepare(context, call)
    call["args"]["params"]["calculator"] = {"calculator_type": "mace_mp"}
    with pytest.raises(PolicyDenied, match="not permitted"):
        prepare(context, call)
    call["args"]["params"]["calculator"] = {"calculator_type": "emt", "command": "rm -rf /"}
    with pytest.raises(PolicyDenied, match="operator-controlled"):
        prepare(context, call)
    defaulted = make_context(tmp_path / "default", default_calculator={"calculator_type": "emt"})
    call["args"]["params"].pop("calculator")
    assert prepare(defaulted, call).calculator["type"] == "emt"


def test_mace_binds_installed_weights_under_model_roots(tmp_path):
    pytest.importorskip("mace")
    roots = tmp_path / "models"
    roots.mkdir()
    weights = roots / "mace-small.model"
    weights.write_bytes(b"weights")
    outside = tmp_path / "elsewhere.model"
    outside.write_bytes(b"weights")
    context = make_context(tmp_path, calculators={"emt", "mace_mp"}, model_roots=(roots,),
                           model_resources={"mace_mp:small": weights, "mace_mp:default": outside})
    call = run_ase_call()
    call["args"]["params"]["calculator"] = {"calculator_type": "mace_mp", "model": "small"}
    action = prepare(context, call)
    assert action.arguments["params"]["calculator"]["model"] == str(weights.resolve())
    assert action.resources == [{"path": str(weights.resolve()), "role": "weights",
                                 "sha256": action.resources[0]["sha256"]}]
    weights.write_bytes(b"replaced")
    with pytest.raises(PreparationError, match="resource changed"):
        validate(context, call, action)
    call["args"]["params"]["calculator"] = {"calculator_type": "mace_mp"}
    with pytest.raises(PolicyDenied, match="model roots"):
        prepare(context, call)
    call["args"]["params"]["calculator"] = {"calculator_type": "mace_mp", "model": "large"}
    with pytest.raises(PolicyDenied, match="No installed model"):
        prepare(context, call)


def test_policy_change_invalidates_preparation(tmp_path):
    context = make_context(tmp_path)
    action = prepare(context, run_ase_call())
    context.policy = ExecutionPolicy(calculators={"emt"}, limits={"max_atoms": 10})
    with pytest.raises(PreparationError, match="policy"):
        validate(context, run_ase_call(), action)


def test_policy_accepts_only_prepared_surface():
    with pytest.raises(ValueError, match="tools"):
        ExecutionPolicy(tools={"run_ase", "run_python"})
    with pytest.raises(ValueError, match="calculators"):
        ExecutionPolicy(calculators={"orca"})
    with pytest.raises(ValueError, match="limits"):
        ExecutionPolicy(limits={"max_steps": 0})
    assert ExecutionPolicy(model_roots=(Path("."),)).identity == ExecutionPolicy(
        model_roots=(str(Path(".").resolve()),)).identity


def test_scoped_paths_stay_in_workspace(tmp_path):
    context = make_context(tmp_path)
    (context.workspace / "link").symlink_to(tmp_path)
    with run_context(context):
        assert scoped_path("out/a.json") == str(context.workspace / "out" / "a.json")
        for path in ("../x", "/etc/passwd", "link/x", ".chemgraph-artifacts/a"):
            with pytest.raises((PermissionError, ValueError)):
                scoped_path(path)
    assert scoped_path("../x") is None
