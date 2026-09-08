"""Default calculator behavior on core-only and Polar-enabled installations."""

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import toml

from chemgraph.schemas.calculators import mace_calc
from chemgraph.utils import calculator_defaults


@pytest.mark.parametrize("polar", [False, True])
def test_defaults_agree_at_initialization(polar, tmp_path):
    root = Path(__file__).resolve().parents[1]
    shipped_text = (root / "config.toml").read_text(encoding="utf-8")
    assert "default" not in toml.loads(shipped_text)["chemistry"]["calculators"]
    config_path = tmp_path / "config.toml"
    config_path.write_text(shipped_text, encoding="utf-8")
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
from ui.config import get_default_config, load_config, resolve_default_calculator
schemas = [ASEInputSchema, ase_input_schema_ensemble]
params = [ASEInputSchema(input_structure_file="water.xyz"), ase_input_schema_ensemble()]
print(json.dumps({{
    "types": [MaceCalc().calculator_type, *[p.calculator.calculator_type for p in params]],
    "model": MaceCalc().get_model_name_for_output(),
    "context": get_calculator_selection_context(),
    "descriptions": [s.model_json_schema()["properties"]["calculator"]["description"] for s in schemas],
    "ui": resolve_default_calculator(get_default_config()),
    "shipped": resolve_default_calculator(load_config(sys.argv[1])),
    "mace_schema": MaceCalc.model_json_schema()["properties"]["calculator_type"],
}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", "import sys\n" + source, str(config_path)], cwd=root,
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout.splitlines()[-1])
    expected_type = "mace_polar" if polar else "mace_mp"
    expected_model = "polar-1-m" if polar else "medium-mpa-0"
    assert data["types"] == [expected_type] * 3
    assert data["model"] == expected_model
    assert data["ui"] == data["shipped"] == expected_type
    assert data["mace_schema"]["default"] == expected_type
    availability = "installed" if polar else "not installed"
    assert f"add-on is {availability}" in data["mace_schema"]["description"]
    assert "before calling run_ase" in data["context"]
    for description in [data["context"], *data["descriptions"]]:
        assert f"calculator_type={expected_type!r}" in description
        assert f"model={expected_model!r}" in description


@pytest.mark.parametrize("polar", [False, True])
def test_explicit_mace_selection_and_ui_configuration_are_preserved(monkeypatch, tmp_path, polar):
    from ui.config import load_config

    monkeypatch.setattr(calculator_defaults, "mace_polar_available", lambda: polar)
    assert mace_calc.MaceCalc().calculator_type == ("mace_polar" if polar else "mace_mp")
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
    monkeypatch.setattr(calculator_defaults, "mace_polar_available", lambda: False)
    with pytest.raises(ImportError, match="requirements/mace-polar.txt"):
        mace_calc.MaceCalc(calculator_type="mace_polar").get_calculator()
    loader.assert_not_called()


def test_default_mp_preserves_upstream_model_default(monkeypatch):
    import mace.calculators

    loader = Mock(return_value=object())
    monkeypatch.setattr(mace.calculators, "mace_mp", loader)
    monkeypatch.setattr(calculator_defaults, "mace_polar_available", lambda: False)
    calc = mace_calc.MaceCalc()
    assert calc.get_calculator() is loader.return_value
    assert loader.call_args.kwargs["model"] is None
    assert calc.get_model_name_for_output() == "medium-mpa-0"


@pytest.mark.parametrize("error", [ImportError, ModuleNotFoundError, ValueError])
def test_unreliable_polar_probe_uses_mp(monkeypatch, error):
    from ui.config import get_default_config, resolve_default_calculator

    monkeypatch.setattr(calculator_defaults.importlib.util, "find_spec", Mock(side_effect=error))
    assert mace_calc.MaceCalc().calculator_type == "mace_mp"
    assert resolve_default_calculator(get_default_config()) == "mace_mp"


def test_polar_module_without_spec_does_not_break_defaults(monkeypatch):
    monkeypatch.setitem(sys.modules, "graph_longrange", ModuleType("graph_longrange"))
    assert mace_calc.MaceCalc().calculator_type == "mace_mp"


def test_saving_unrelated_settings_preserves_automatic_selection(monkeypatch, tmp_path):
    from ui.config import load_config, resolve_default_calculator, save_config

    path = tmp_path / "config.toml"
    monkeypatch.setattr(calculator_defaults, "mace_polar_available", lambda: False)
    config = load_config(str(path))
    assert resolve_default_calculator(config) == "mace_mp"
    config["api"]["argo"]["argo_user"] = "test-user"
    assert save_config(config, str(path))
    assert "default" not in toml.load(path)["chemistry"]["calculators"]
    monkeypatch.setattr(calculator_defaults, "mace_polar_available", lambda: True)
    reloaded = load_config(str(path))
    assert resolve_default_calculator(reloaded) == "mace_polar"
    assert reloaded["api"]["argo"]["argo_user"] == "test-user"
    reloaded["chemistry"]["calculators"]["default"] = "mace_mp"
    assert save_config(reloaded, str(path))
    assert resolve_default_calculator(load_config(str(path))) == "mace_mp"


def test_configuration_recovery_does_not_import_torch(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text("[broken", encoding="utf-8")
    source = '''
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise OSError("simulated broken Torch library")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from ui.config import get_default_config, load_config, resolve_default_calculator
assert load_config(sys.argv[1]) == get_default_config()
assert resolve_default_calculator(get_default_config()) in {"mace_mp", "mace_polar"}
assert "torch" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", source, str(path)], check=True, capture_output=True)
