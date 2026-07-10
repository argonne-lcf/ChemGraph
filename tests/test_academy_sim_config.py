from __future__ import annotations

import json

import pytest

from chemgraph.academy_sim.config import load_config


def test_load_config_resolves_graphs_and_paths(tmp_path):
    config_path = tmp_path / "config.jsonc"
    config_path.write_text(
        """
        {
          // JSONC comments are accepted.
          "run_id": "run-1",
          "task": "test",
          "initial_graph": "planner",
          "artifacts": {"run_dir": "runs"},
          "model": {"config_file": "lm.json"},
          "graphs": {
            "planner": {
              "workflow_type": "single_agent",
              "allowed_peers": ["executor"],
              "science_tools": []
            },
            "executor": {
              "workflow_type": "single_agent",
              "allowed_peers": ["planner"],
              "science_tools": [
                {
                  "name": "general",
                  "transport": "streamable_http",
                  "url": "http://127.0.0.1:9003/mcp/"
                }
              ]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.graph("planner").name == "planner"
    assert config.initial_graph == "planner"
    assert config.graph("executor").science_tools[0].name == "general"
    assert config.run_dir() == tmp_path / "runs" / "run-1"
    assert config.model_for_graph("planner").config_file == tmp_path / "lm.json"


def test_config_rejects_unknown_peer(tmp_path):
    config_path = tmp_path / "config.jsonc"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task": "test",
                "model": {"config_file": "lm.json"},
                "graphs": {
                    "planner": {
                        "allowed_peers": ["missing"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unknown allowed_peers"):
        load_config(config_path)


def test_config_accepts_http_exchange_registration(tmp_path):
    config_path = tmp_path / "config.jsonc"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task": "test",
                "initial_graph": "planner",
                "bootstrap_mode": "manual",
                "exchange": {
                    "type": "http",
                    "registration": "exchange",
                    "url": "https://exchange.example/v1",
                },
                "model": {"config_file": "lm.json"},
                "graphs": {
                    "planner": {
                        "allowed_peers": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.exchange.type == "http"
    assert config.exchange.registration == "exchange"
    assert config.bootstrap_mode == "manual"


def test_config_maps_globus_exchange_alias_to_http(tmp_path):
    config_path = tmp_path / "config.jsonc"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task": "test",
                "exchange": {"type": "globus"},
                "model": {"config_file": "lm.json"},
                "graphs": {"planner": {}},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.exchange.type == "http"
