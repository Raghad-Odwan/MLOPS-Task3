"""
Loads the project configuration from config/config.yaml.
Every other module gets its paths and parameters from here,
instead of hardcoding them.
"""

import yaml
from pathlib import Path

# Project root is two levels up from this file (src/config_loader.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Read the YAML config file and return it as a dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(relative_path: str) -> Path:
    """Turn a path from the config file into a full path from the project root."""
    return PROJECT_ROOT / relative_path
