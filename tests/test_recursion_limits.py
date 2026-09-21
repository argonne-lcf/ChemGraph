"""Default graph budgets agree across persisted, evaluation, and UI configs."""

import pytest
import toml
from pydantic import ValidationError

from chemgraph.eval.cli import build_config_from_args, parse_args
from chemgraph.eval.config import BenchmarkConfig
from chemgraph.memory.schemas import MainAgentGraphConfig
from ui.config import merge_config_defaults


@pytest.mark.parametrize("options,expected", [({}, 200), ({"recursion_limit": 17}, 17)])
def test_config_defaults_preserve_explicit_recursion_limits(tmp_path, options, expected):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    benchmark = BenchmarkConfig(
        models=["fake"], dataset=str(dataset), judge_type="structured", **options
    )
    graph_config = MainAgentGraphConfig(model_name="fake", **options)
    restored = MainAgentGraphConfig.model_validate_json(graph_config.model_dump_json())
    ui_config = merge_config_defaults({"general": options})

    assert benchmark.recursion_limit == expected
    assert restored.recursion_limit == expected
    assert ui_config["general"]["recursion_limit"] == expected


@pytest.mark.parametrize("profile_limit", [None, {}, 17])
@pytest.mark.parametrize("limit", [None, 0, -1, 33, 200])
def test_evaluation_recursion_limit_defaults_overrides_and_validation(
    tmp_path, profile_limit, limit,
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    argv = ["--models", "fake", "--dataset", str(dataset), "--judge-type", "structured"]
    if profile_limit is not None:
        profile = {} if profile_limit == {} else {"recursion_limit": profile_limit}
        config = tmp_path / "config.toml"
        config.write_text(toml.dumps({"eval": {"profiles": {"test": profile}}}))
        argv += ["--config", str(config), "--profile", "test"]
    if limit is not None:
        argv += ["--recursion-limit", str(limit)]
    args = parse_args(argv)
    if limit is not None and limit < 1:
        with pytest.raises(ValidationError, match="recursion_limit"):
            build_config_from_args(args)
    else:
        expected = limit if limit is not None else (profile_limit or 200)
        assert build_config_from_args(args).recursion_limit == expected


@pytest.mark.parametrize("limit", [0, -1])
def test_evaluation_rejects_invalid_stored_and_direct_limits(tmp_path, limit):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    options = {"dataset": str(dataset), "judge_type": "structured", "recursion_limit": limit}
    with pytest.raises(ValidationError, match="recursion_limit"):
        BenchmarkConfig(models=["fake"], **options)
    config = tmp_path / "config.toml"
    config.write_text(toml.dumps({"eval": {"profiles": {"test": options}}}))
    with pytest.raises(ValidationError, match="recursion_limit"):
        BenchmarkConfig.from_profile("test", models=["fake"], config_file=str(config))
