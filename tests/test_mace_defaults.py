"""Default calculator behavior on core-only and Polar-enabled installations."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from chemgraph.schemas.calculators import mace_calc


@pytest.mark.parametrize("polar", [False, True])
def test_defaults_agree_at_initialization(polar):
    # Import in a fresh process so schema descriptions see the same installation
    # state as the default factories, without changing other tests' model classes.
    source = f'''
import importlib.util
import json
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: (
    object() if {polar!r} else None
) if name == "graph_longrange" else find_spec(name, *a, **kw)
from chemgraph.schemas.ase_input import (
    ASEInputSchema, ase_input_schema_ensemble, get_calculator_selection_context,
)
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from ui.config import get_default_config, load_config
schemas = [ASEInputSchema, ase_input_schema_ensemble]
params = [ASEInputSchema(input_structure_file="water.xyz"), ase_input_schema_ensemble()]
print(json.dumps({{
    "types": [MaceCalc().calculator_type, *[p.calculator.calculator_type for p in params]],
    "model": MaceCalc().get_model_name_for_output(),
    "context": get_calculator_selection_context(),
    "descriptions": [s.model_json_schema()["properties"]["calculator"]["description"] for s in schemas],
    "ui": get_default_config()["chemistry"]["calculators"]["default"],
    "shipped": load_config("config.toml")["chemistry"]["calculators"]["default"],
}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout.splitlines()[-1])
    expected_type = "mace_polar" if polar else "mace_mp"
    expected_model = "polar-1-m" if polar else "medium-mpa-0"
    assert data["types"] == [expected_type] * 3
    assert data["model"] == expected_model
    assert data["ui"] == data["shipped"] == expected_type
    for description in [data["context"], *data["descriptions"]]:
        assert f"calculator_type={expected_type!r}" in description
        assert f"model={expected_model!r}" in description


@pytest.mark.parametrize("polar", [False, True])
def test_explicit_mace_selection_and_ui_configuration_are_preserved(monkeypatch, tmp_path, polar):
    from ui.config import load_config

    monkeypatch.setattr(mace_calc, "mace_polar_available", lambda: polar)
    for variant in ("mace_polar", "mace_mp", "mace_off", "mace_anicc"):
        calc = mace_calc.MaceCalc(calculator_type=variant, model="/models/custom.model")
        assert calc.calculator_type == variant
        assert calc.get_model_name_for_output() == "/models/custom.model"
    config = tmp_path / "explicit.toml"
    config.write_text('[chemistry.calculators]\ndefault = "mace_polar"\n')
    assert load_config(str(config))["chemistry"]["calculators"]["default"] == "mace_polar"


def test_missing_polar_fails_before_loader(monkeypatch):
    import mace.calculators

    loader = Mock(side_effect=AssertionError("must not load or download weights"))
    monkeypatch.setattr(mace.calculators, "mace_polar", loader)
    monkeypatch.setattr(mace_calc, "mace_polar_available", lambda: False)
    with pytest.raises(ImportError, match="requirements/mace-polar.txt"):
        mace_calc.MaceCalc(calculator_type="mace_polar").get_calculator()
    loader.assert_not_called()


def test_default_mp_preserves_upstream_model_default(monkeypatch):
    import mace.calculators

    loader = Mock(return_value=object())
    monkeypatch.setattr(mace.calculators, "mace_mp", loader)
    monkeypatch.setattr(mace_calc, "mace_polar_available", lambda: False)
    calc = mace_calc.MaceCalc()
    assert calc.get_calculator() is loader.return_value
    assert loader.call_args.kwargs["model"] is None
    assert calc.get_model_name_for_output() == "medium-mpa-0"
