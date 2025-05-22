"""
Configuration utilities for Sifaka.

This module provides functions for loading and saving configuration.
"""

import json
import os
from typing import Any, Dict, Optional

from sifaka.errors import ConfigurationError


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a file.

    Args:
        path: Path to the configuration file. If None, looks for config in
            standard locations.

    Returns:
        The configuration as a dictionary.

    Raises:
        ConfigurationError: If the configuration file cannot be loaded.
    """
    # Default config locations
    default_locations = [
        os.path.expanduser("~/.sifaka/config.json"),
        os.path.join(os.getcwd(), "sifaka_config.json"),
    ]

    # Use provided path or try default locations
    paths_to_try = [path] if path else default_locations

    for config_path in paths_to_try:
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return dict(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                raise ConfigurationError(f"Failed to load config from {config_path}: {e}")

    # Return empty config if no config file found
    return {}


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to a file.

    Args:
        config: The configuration to save.
        path: Path to the configuration file.

    Raises:
        ConfigurationError: If the configuration cannot be saved.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write config to file
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    except (IOError, OSError) as e:
        raise ConfigurationError(f"Failed to save config to {path}: {e}")
