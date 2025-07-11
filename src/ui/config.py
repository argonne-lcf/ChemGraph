"""
Configuration management for ChemGraph Streamlit app.
"""

import toml
import os
from typing import Dict, Any


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = toml.load(f)
                # Validate configuration structure
                default_config = get_default_config()

                # Ensure all required sections exist
                for section in ["general", "llm", "api", "chemistry", "output"]:
                    if section not in config:
                        config[section] = default_config[section]
                    elif isinstance(config[section], dict) and isinstance(
                        default_config[section], dict
                    ):
                        # Merge missing keys from default
                        for key, value in default_config[section].items():
                            if key not in config[section]:
                                config[section][key] = value

                return config
        else:
            # Create default configuration file if it doesn't exist
            default_config = get_default_config()
            save_config(default_config, config_path)
            return default_config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return get_default_config()


def save_config(config: Dict[str, Any], config_path: str = "config.toml") -> bool:
    """Save configuration to TOML file."""
    try:
        with open(config_path, "w") as f:
            toml.dump(config, f)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "general": {
            "model": "gpt-4o-mini",
            "workflow": "single_agent",
            "output": "state",
            "structured": False,
            "report": True,
            "thread": 1,
            "recursion_limit": 20,
            "verbose": False,
        },
        "llm": {
            "temperature": 0.1,
            "max_tokens": 4000,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "api": {
            "openai": {"base_url": "https://api.openai.com/v1", "timeout": 30},
            "anthropic": {"base_url": "https://api.anthropic.com", "timeout": 30},
            "google": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "timeout": 30,
            },
            "local": {"base_url": "http://localhost:11434", "timeout": 60},
        },
        "chemistry": {
            "optimization": {"method": "BFGS", "fmax": 0.05, "steps": 200},
            "calculators": {"default": "mace_mp", "fallback": "emt"},
        },
        "output": {
            "files": {
                "directory": "./chemgraph_output",
                "formats": ["xyz", "json", "html"],
            },
            "visualization": {"enable_3d": True, "viewer": "py3dmol"},
        },
    }


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested configuration for easier access."""
    flattened = {}

    # Handle general settings
    if "general" in config:
        flattened.update(config["general"])

    # Handle other sections
    for section in ["llm", "api", "chemistry", "output"]:
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened[f"{section}_{key}_{subkey}"] = subvalue
                else:
                    flattened[f"{section}_{key}"] = value

    return flattened
