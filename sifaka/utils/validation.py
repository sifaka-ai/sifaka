"""Validation utilities."""

from typing import Dict, Any


def validate_config(config: Dict[str, Any]) -> None:
    """Validate a configuration dictionary.

    Args:
        config: The configuration dictionary to validate.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    if not config.get("name"):
        raise ValueError("Configuration must have a name")

    if not config.get("description"):
        raise ValueError("Configuration must have a description")

    if "params" in config and not isinstance(config["params"], dict):
        raise ValueError("Parameters must be a dictionary")