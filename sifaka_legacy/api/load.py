"""
Load Module

This module provides functions for loading chains and models from configurations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .chain import Chain
from .factory import model

logger = logging.getLogger(__name__)


def load_chain(
    config: Union[Dict[str, Any], str, Path], model_provider: Optional[Any] = None
) -> Chain:
    """
    Load a chain from a configuration.

    Args:
        config: Either a dict containing the configuration, a string with JSON content,
               or a path to a JSON file
        model_provider: An optional model provider to use for the chain

    Returns:
        A configured Chain instance

    Raises:
        ValueError: If the configuration can't be loaded
    """
    # Load the configuration
    if isinstance(config, (str, Path)):
        # Check if it's a file path
        path = Path(config)
        if path.exists() and path.is_file():
            with open(path, "r") as f:
                config_dict = json.load(f)
        else:
            # Try to parse as JSON string
            try:
                config_dict = json.loads(config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid configuration: {e}")
    else:
        config_dict = config

    # Create the chain
    chain = Chain.from_dict(config_dict)

    # Set the model provider if provided
    if model_provider:
        chain.set_model(model_provider)

    return chain


def load_model(config: Union[Dict[str, Any], str, Path]) -> Any:
    """
    Load a model from a configuration.

    Args:
        config: Either a dict containing the configuration, a string with JSON content,
               or a path to a JSON file

    Returns:
        A configured model provider

    Raises:
        ValueError: If the configuration can't be loaded
    """
    # Load the configuration
    if isinstance(config, (str, Path)):
        # Check if it's a file path
        path = Path(config)
        if path.exists() and path.is_file():
            with open(path, "r") as f:
                config_dict = json.load(f)
        else:
            # Try to parse as JSON string
            try:
                config_dict = json.loads(config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid configuration: {e}")
    else:
        config_dict = config

    # Get the model type and config
    model_type = config_dict.get("type")
    model_config = config_dict.get("config", {})

    if not model_type:
        raise ValueError("Model configuration must include a 'type' field")

    # Create the model
    return model(model_type, **model_config)
