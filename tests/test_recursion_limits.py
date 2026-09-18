"""Default graph budgets agree across persisted, evaluation, and UI configs."""

import pytest

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
