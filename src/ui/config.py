"""
Configuration management for ChemGraph Streamlit app.
"""

import copy
import toml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from chemgraph.utils.config_utils import flatten_config as _flatten_config
from chemgraph.utils.calculator_defaults import get_default_mace_calculator_type

CONFIG_PATH_ENV = "CHEMGRAPH_CONFIG"


def _source_checkout_root() -> Optional[Path]:
    """Return the repository root when running from a source checkout.

    ``src/ui/config.py`` sits two levels below the repository root, which is
    identified by its ``pyproject.toml``.  In a wheel installation the same
    ancestor is ``lib/pythonX.Y``, which has no such marker and is frequently
    read-only.
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def user_config_dir() -> Path:
    """Return the per-user directory for an installed ChemGraph's config."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.environ.get("APPDATA")
    base_path = Path(base) if base else Path.home() / ".config"
    return base_path / "chemgraph"


def default_config_path() -> str:
    """Resolve where ``config.toml`` lives for this installation.

    Precedence: the ``CHEMGRAPH_CONFIG`` environment variable, then
    ``<repo>/config.toml`` for a source checkout (the directory the app is
    meant to be launched from per the README, rather than the current working
    directory), then a writable per-user location for wheel installations.
    Resolving relative to the package for installed copies placed the file
    inside ``lib/pythonX.Y/``; saves there fail silently on read-only
    installs and the settings are gone on the next start.
    """
    override = os.environ.get(CONFIG_PATH_ENV)
    if override:
        return str(Path(override).expanduser())
    checkout = _source_checkout_root()
    if checkout is not None:
        return str(checkout / "config.toml")
    return str(user_config_dir() / "config.toml")


_DEFAULT_CONFIG_PATH = default_config_path()


def config_path() -> str:
    """Return the active default configuration file path."""
    return _DEFAULT_CONFIG_PATH


def merge_config_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing settings without resolving automatic calculator selection."""
    config = copy.deepcopy(config)
    default_config = get_default_config()
    for section in ["general", "api", "chemistry", "output"]:
        if section not in config:
            config[section] = default_config[section]
        elif isinstance(config[section], dict):
            for key, value in default_config[section].items():
                if key not in config[section]:
                    if section == "api" and key in {"argo", "vllm"}:
                        continue
                    config[section][key] = value
                elif isinstance(config[section][key], dict) and isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        config[section][key].setdefault(subkey, subvalue)
    return config


def resolve_default_calculator(config: Dict[str, Any]) -> str:
    """Resolve a calculator for display/use without persisting the detection."""
    calculators = config.get("chemistry", {}).get("calculators", {})
    if "default" in calculators:
        return calculators["default"]
    return get_default_mace_calculator_type()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a TOML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the TOML configuration file.

    Returns
    -------
    dict[str, Any]
        Nested configuration dictionary with defaults filled in.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return merge_config_defaults(toml.load(f))
        else:
            # Create default configuration file if it doesn't exist
            default_config = get_default_config()
            save_config(default_config, config_path)
            return default_config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return get_default_config()


#: Human-readable reason for the most recent failed ``save_config`` call, or
#: ``None`` after a successful save. UI pages surface it to the user.
last_save_error: Optional[str] = None


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """Save configuration to a TOML file.

    Parameters
    ----------
    config : dict[str, Any]
        Nested configuration dictionary to write.
    config_path : str, optional
        Destination TOML file path.

    Returns
    -------
    bool
        ``True`` if the file was written successfully.
    """
    global last_save_error
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    try:
        parent = os.path.dirname(os.path.abspath(config_path))
        os.makedirs(parent, exist_ok=True)
        with open(config_path, "w") as f:
            toml.dump(config, f)
        last_save_error = None
        return True
    except Exception as e:
        last_save_error = f"{config_path}: {e}"
        print(f"Error saving configuration to {last_save_error}")
        return False


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "general": {
            "model": "gpt-4o-mini",
            "workflow": "single_agent",
            "output": "state",
            "structured": False,
            "report": False,
            "thread": 1,
            "recursion_limit": 20,
            "human_supervised": False,
            "verbose": False,
        },
        "api": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "timeout": 30,
            },
            "argo": {
                "base_url": "https://apps.inside.anl.gov/argoapi/v1",
                "argo_user": "",
            },
            "vllm": {"base_url": ""},
            "anthropic": {"base_url": "https://api.anthropic.com", "timeout": 30},
            "google": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "timeout": 30,
            },
            "alcf": {
                "base_url": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
                "timeout": 30,
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": 60,
            },
            "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "timeout": 30,
            },
            "local": {"base_url": "http://localhost:11434", "timeout": 60},
        },
        "chemistry": {
            "optimization": {"method": "BFGS", "fmax": 0.05, "steps": 200},
            "calculators": {"fallback": "emt"},
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
    """Flatten nested configuration for easier access.

    Parameters
    ----------
    config : dict[str, Any]
        Nested configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Flattened configuration dictionary.
    """
    return _flatten_config(config)
